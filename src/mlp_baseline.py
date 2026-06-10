import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout_p = dropout
        self.layers = ModuleList()

        if num_layers == 1:
            self.layers.append(Linear(in_channels, out_channels))
        else:
            self.layers.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(Linear(hidden_channels, hidden_channels))
            self.layers.append(Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x
