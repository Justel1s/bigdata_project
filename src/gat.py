import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=8):
        super().__init__()
        self.dropout_p = dropout
        self.convs = ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels,
                            heads=heads, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels,
                        heads=1, concat=False, dropout=dropout)
            )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def embed(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        return x
