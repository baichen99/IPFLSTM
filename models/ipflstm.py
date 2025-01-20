
import torch
import torch.nn as nn
from informer import Informer

class IPFLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, lstm_dim, lstm_layers, attention_heads):
        super(IPFLSTM, self).__init__()

        # PINN layers
        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())
        self.pinn_layers = nn.Sequential(*layers)

        # LSTM module
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_dim, num_layers=lstm_layers, batch_first=True)

        # Informer module
        self.informer = Informer(
            input_size=lstm_dim,
            output_size=out_dim,
            num_heads=attention_heads,
            seq_len=100,
            pred_len=1
        )

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=-1)
        pinn_output = self.pinn_layers(inputs)
        lstm_output, _ = self.lstm(pinn_output.unsqueeze(0))  # Batch size added for LSTM
        informed_output = self.informer(lstm_output).squeeze(0)
        return informed_output
