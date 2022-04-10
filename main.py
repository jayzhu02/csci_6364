import torch
import torch.nn as nn
from data_loader import ETHDataLoader
from models.LSTM import LSTM
from models.GRU import GRU
from plot import plot_predict_data
import time
import pandas as pd


def train(dataloader, model, epochs=10, learning_rate=0.0001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on : {device}")

    model = model.to(device)
    # Save mean and std of training dataset
    model.mean = dataloader.dataset.mean
    model.std = dataloader.dataset.std

    lossfunc = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    state = None
    print("/----------------------------/")
    print("Start training...")
    for epoch in range(epochs):
        start = time.time()
        for x, y in dataloader.iter():
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y = y.view(-1, y.shape[2])
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            output, state = model(x, state)
            loss = lossfunc(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if epoch % 50 == 0:
            print("/----------------------------/")
            print(f"epoch: {epoch}, loss:{round(loss.cpu().item(), 3)}, time:{round(time.time() - start, 3)}s")

        if epoch % 300 == 0 and epoch != 0:
            print(f"Saving model...")
            torch.save(model, f"./checkpoints/{model.model_type}-epoch-{epoch}-{round(loss.cpu().item(), 3)}.pth")


def predict(model, pre_data, day_pred, device=None):
    """

    :param model:
    :param pre_data: 1 * num_steps * input_dim
    :return:
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on : {device}")

    pre_data = torch.from_numpy(pre_data).float().to(device)
    model = model.eval().to(device)
    pre_len = pre_data.shape[1]
    state = None
    outputs = []
    print("/----------------------------/")
    print("Start predicting...")
    with torch.no_grad():
        for i in range(day_pred + pre_len - 1):
            if i < pre_len - 1:
                x = pre_data.to(device)
            else:
                x = torch.cat([x, out[-1].reshape(1, 1, out.shape[1])], dim=1)

            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].to(device), state[1].to(device))
                else:
                    state = state.to(device)

            out, _ = model(x, state)

    return x.cpu() * model.std + model.mean


def predict_from_csv(model_path, file, num_steps, day_pred, device=None):
    model = torch.load(model_path)
    raw_data = pd.read_csv(file)
    x = raw_data.head(num_steps)[['Open', 'High', 'Low', 'Close', 'Volume']]
    x = x.values.reshape(1, x.shape[0], x.shape[1])
    x = (x - model.mean) / model.std
    y = raw_data[:num_steps + day_pred]
    pred_y = predict(model, x, day_pred, device)

    # test_loader = ETHDataLoader(file, batch_size=1, num_steps=num_steps, istrain=False)
    # for x, y in test_loader.iter():
    #     x = (x - model.mean) / model.std
    #     pred_y = predict(model, x, day_pred, device)
    #     break
    plot_predict_data(y, pred_y[:, -day_pred:, :], 'LSTM_predict_data')
    return y, pred_y[:, -day_pred:, :]


def main(mode="train"):
    batch_size = 4
    num_steps = 30
    day_to_pred = 7

    if mode == "train":
        data_loader = ETHDataLoader('./data/ETH_USD-3_year.csv', batch_size=batch_size, num_steps=num_steps)
        # model = LSTM(input_size=5, hidden_size=256, output_size=5)
        model = GRU(input_size=5, hidden_size=128, output_size=5)
        train(data_loader, model, epochs=1000)
    else:
        res, pred_data = predict_from_csv('./checkpoints/LSTM-epoch-900-0.008.pth', './data/ETH_USD-1_year.csv',
                                          num_steps, day_to_pred)


if __name__ == '__main__':
    main(mode='test')
