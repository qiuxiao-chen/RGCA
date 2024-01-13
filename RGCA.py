# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from torch import nn

class ResNetForBEVDet_graph_spatial(nn.Module):
    def __init__(self, numC_input, node=8):
        super().__init__()

        self.shrink_origin = nn.Conv2d(numC_input, numC_input, kernel_size=1,stride=16)

        self.shrink_graph = nn.Conv2d(numC_input, numC_input, kernel_size=1,stride=16)
        self.graph_1 = nn.Conv1d(node*node, node*node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.graph_2 = nn.Conv1d(numC_input, numC_input, kernel_size=1)

        self.extend = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        

    def forward(self, x):
        n, channel, h, w = x.shape
        shrink_origin = self.shrink_origin(x)
        h_shrink, w_shrink = shrink_origin.shape[2:]
        
        shrink_graph = self.shrink_graph(x).view(n,channel,-1)
        shrink_graph = self.graph_1(shrink_graph.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        shrink_graph = self.graph_2(self.relu(shrink_graph)).view(n,channel,h_shrink,-1)

        shrink = shrink_origin + shrink_graph

        shrink_extend = self.extend(shrink)

        x_tmp = torch.cat((x,shrink_extend),1)

        return x_tmp
    
temp = torch.randn((8, 64, 128, 128))    # [batch_size,channel,height,weight]
conv = ResNetForBEVDet_graph_spatial(64)
print(conv(temp).size())