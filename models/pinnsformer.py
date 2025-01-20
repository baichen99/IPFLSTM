
import torch
import torch.nn as nn
from informer import Informer  # Assuming a third-party Informer library is installed

class PINNsFormer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, attention_heads):
        super(PINNsFormer, self).__init__()

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

        # Informer module
        self.informer = Informer(
            input_size=hidden_dim,
            output_size=out_dim,
            num_heads=attention_heads,
            seq_len=100,
            pred_len=1
        )

    def forward(self, x, y, t):
        inputs = torch.cat((x, y, t), dim=-1)
        pinn_output = self.pinn_layers(inputs)
        informed_output = self.informer(pinn_output.unsqueeze(0)).squeeze(0)  # Batch size added for Informer
        return informed_output
