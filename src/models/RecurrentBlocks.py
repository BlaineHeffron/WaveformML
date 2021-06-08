from torch import nn, zeros

from src.models.ConvBlocks import LinearBlock


class RecurrentBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, nonlinearity='relu', bias=True, dropout=0., bidirectional=False):
        super(RecurrentBlock, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, nonlinearity=nonlinearity, bias=bias, dropout=dropout,
                          bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        return self.rnn(x, hidden)

    def init_hidden(self, batch_size):
        hidden = zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden


class RecurrentNet(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, n_layers, n_lin, out_size, nonlinearity='relu', bias=True, dropout=0., bidirectional=False):
        super(RecurrentNet, self).__init__()
        self.rnn_block = RecurrentBlock(input_size, hidden_size, n_layers, nonlinearity=nonlinearity, bias=bias, dropout=dropout,
                          bidirectional=bidirectional)
        if n_lin > 0:
            self.linear = LinearBlock(hidden_size*seq_len, out_size, n_lin).func
        else:
            self.linear = None
        self.out_size = out_size
        self.flatten = nn.Flatten()

    def forward(self, x):
        out, hidden = self.rnn_block(x)
        out = self.flatten(out)
        if self.linear:
            return self.linear(out)
        else:
            if self.out_size == 1:
                return out[:, -1]
            else:
                raise IOError("must have n_lin > 0 if out_size is > 1")
