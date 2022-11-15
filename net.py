import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256]), k, aggr)
        self.lin1 = Linear(256 + 128 + 64 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)
        out = self.lin1(torch.cat([x1, x2, x3, x4], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)