import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.model_type = "GRU"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state = None

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        y, self.state = self.gru(x, state)  # batch * num_steps * hid_dim
        out = self.dense(y.reshape(-1, y.shape[-1]))  # batch * num_steps , out_dim
        return out, self.state
