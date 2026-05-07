import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bi_LSTM(nn.module):

    def __init__(
        self, input_size=3, hidden_size=512, num_layers=2, batch_size=64, seq_len=120, bidirectional=True, dropout=0.5
    ):

        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_directions = num_directions

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)
        return h0, c0

    def forward(self, input_tensor):
        ## Create random input tensor
        # input_tensor = torch.randn(seq_len, batch_size, input_size)
        batch_size = input_tensor.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        output, (h_n, c_n) = self.lstm(input_tensor, (h_0, c_0))
        out = self.output_layer(output)
        return out, (h_n, c_n)
