import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        output_size,
        n_layers = 4,
        dropout_p = .2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.rnn = nn.LSTM(
            input_size= input_size,
            hidden_size = hidden_size,
            num_layers= n_layers,
            batch_first= True,
            dropout=dropout_p,
            bidirectional= True
        )

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size *2, output_size),
            #nn.Softmax(dim=-1)
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        print(x.shape)
        z,_ = self.rnn(x)
        z = z[:, -1]
        y = self.layers(z)
        return y