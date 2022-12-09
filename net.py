import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool


class ClassifierNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256]), k, aggr)
        self.lin1 = Linear(256 + 128 + 64 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.2, norm=None)

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


class WrapSegmentationNet(torch.nn.Module):
    def __init__(self, in_hits, in_channels, out_channels, path='/data/rradev/supernovae_trigger/outputs/new_segment_epoch12_acc_0.98.pth'):
        super().__init__()
        self.segmentation_network = self.load_pretrained_model(path, out_channels)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_hits * in_channels, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_channels)
        )

    def load_pretrained_model(self, path, out_channels):
        model_state_dict = torch.load(path)['model_state_dict']
        segmentation_net = SegmentationNet(7, out_channels)
        segmentation_net.load_state_dict(model_state_dict)
        for param in segmentation_net.parameters():
            param.requires_grad = False
        return segmentation_net
        
    def forward(self, data, batch_size):
        x = self.segmentation_network(data)
        
        # Pass output of segmentation network to MLP
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        x = F.log_softmax(x, dim=1)
        return x


class SegmentationNet(torch.nn.Module):
    def __init__(self,in_channels, out_channels, k=30, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * in_channels, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        self.mlp = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5,
                       norm=None)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)