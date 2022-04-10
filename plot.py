import matplotlib.pyplot as plt
import pandas as pd

def plot_history_data(file):
    data = pd.read_csv(file)
    time = data['Date'].tolist()
    high = data['High'].tolist()
    plt.figure(figsize=(12, 8))
    plt.xticks(range(0, len(time), 180))
    plt.title('History High data of ETH')
    plt.xlabel('Date')
    plt.ylabel('Highest Price')
    plt.plot(time, high)
    plt.savefig('./img/history_data.jpg')

    plt.show()


def plot_predict_data(y, pred_y, file_name):
    time = y['Date'].tolist()
    high = y['High'].tolist()
    pred_h = pred_y[:, :, 1].reshape(-1)
    plt.figure(figsize=(12, 8))
    plt.xticks(range(0, len(time), 10))
    plt.title('Predict High data of ETH')
    plt.xlabel('Date')
    plt.ylabel('Highest Price')
    plt.plot(time, high, 'b', label='real_data')
    plt.plot(time[-pred_h.shape[0]:], pred_h, 'r', label='predict_data')
    plt.legend()
    plt.savefig(f'./img/{file_name}.jpg')
    plt.show()


# plot_history_data('./data/ETH_USD-3_year.csv')