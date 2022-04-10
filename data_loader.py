import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class ETHDataset(Dataset):
    def __init__(self, file_path, istrain):
        self.data = read_csv(file_path)
        self.istrain = istrain
        if istrain:
            # input normalization
            scaler = StandardScaler()
            scaler.fit(self.data)
            self.data = scaler.transform(self.data)
            # mean and variance of training set
            self.mean = scaler.mean_
            self.std = np.sqrt(scaler.var_)
        else:
            self.mean = 0
            self.std = 0


    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class ETHDataLoader(DataLoader):
    def __init__(self, filepath, batch_size, num_steps, istrain=True, shuffle=False):
        self.filepath = filepath
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.dataset = ETHDataset(self.filepath, istrain)

    def iter(self):
        # Need to use consecutive sampling to generate label
        input, label = self.dataset[:-1, :], self.dataset[1:, :]

        data_len = len(input)
        batch_len = data_len // self.batch_size
        input = input[0:self.batch_size * batch_len, :].reshape(self.batch_size, batch_len, -1)
        label = label[0:self.batch_size * batch_len, :].reshape(self.batch_size, batch_len, -1)
        epoch_size = (batch_len - 1) // self.num_steps

        for i in range(epoch_size):
            i = i * self.num_steps
            x = input[:, i:i + self.num_steps, :]
            y = label[:, i:i + self.num_steps, :]

            yield x, y


def read_csv(file):
    data = pd.read_csv(file)
    data[data.columns[1:5]] = round(data[data.columns[1:5]], 3)
    data['Volume'] = round(data['Volume'], 1)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    return data


# def collate_fn(data):
#     open, high, low, close, volume, label = zip(*data)
#
#     open = torch.FloatTensor(open).view(-1, 1)
#     high = torch.FloatTensor(high).view(-1, 1)
#     low = torch.FloatTensor(low).view(-1, 1)
#     close = torch.FloatTensor(close).view(-1, 1)
#     volume = torch.IntTensor(volume).view(-1, 1)
#     label = torch.FloatTensor(label).view(-1, 1)
#
#     return torch.cat([open, high, low, close, volume], dim=1), label
