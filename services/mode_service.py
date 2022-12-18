import torch
import torch.nn as nn
import math
import random
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


batch_size = 100
seq_len = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


class StockDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.data = torch.from_numpy(data).float().view(-1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, index):
        return self.data[index : index + self.seq_len], self.data[index + self.seq_len]


def calculate_metrics(model, scalar, data_loader):
    pred_arr = []
    y_arr = []
    with torch.no_grad():
        hn, cn = model.init()
        for _, item in enumerate(data_loader):
            x, y = item
            x, y = x.to(device), y.to(device)
            x = x.view(seq_len, batch_size, 1)
            pred = model(x, hn, cn)[0]
            pred = scalar.inverse_transform(pred.detach().cpu().numpy()).reshape(-1)
            y = scalar.inverse_transform(
                y.detach().cpu().numpy().reshape(1, -1)
            ).reshape(-1)
            pred_arr = pred_arr + list(pred)
            y_arr = y_arr + list(y)

        ind = random.randint(0, 99)
        return math.sqrt(mean_squared_error(y_arr, pred_arr)), pred_arr, y_arr


class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, drop_prob=0.2):
        super(LstmModel, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.dropout(out)

        final_out = self.fc(out[-1])
        return final_out, hn, cn

    def init(self):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0
