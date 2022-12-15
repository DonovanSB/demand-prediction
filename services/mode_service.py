import torch
import torch.nn as nn

batch_size = 100
device = "cuda" if torch.cuda.is_available() else "cpu"


class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, drop_prob=0.3):
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
