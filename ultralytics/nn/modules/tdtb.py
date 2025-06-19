import torch
from torch import nn
import torch.nn.functional as F

from .conv import Conv, DWConv


class Z_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.concat((torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), axis=1)


class TA(nn.Module):
    """ Implementation of the Triplet Attention module as described in the paper:
    "Rotate to Attend: Convolutional Triplet Attention Module" (https://arxiv.org/pdf/2010.03045).
    """
    def __init__(self):
        super().__init__()
        self.z_pool = Z_Pool()
        self.conv_spatial = DWConv(2, 1, k=7, act=nn.Sigmoid())
        self.conv_channel_width = DWConv(2, 1, k=7, act=nn.Sigmoid())
        self.conv_channel_height = DWConv(2, 1, k=7, act=nn.Sigmoid())
        
    def forward(self, x):
        # x: [B, C, H, W]
        # spatial attention
        a1 = self.conv_spatial(self.z_pool(x))
        y1 = x * a1
        # channel-width attention
        a2 = self.conv_channel_width(self.z_pool(x.permute(0, 2, 1, 3))).permute(0, 2, 1, 3)
        y2 = x * a2
        # channel-height attention
        a3 = self.conv_channel_height(self.z_pool(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        y3 = x * a3
        y = (y1 + y2 + y3) / 3
        return y
    

class TDTB(nn.Module):
    """ Implementation of the Triplet Domain Transfer Block (TDTB)
    """
    def __init__(self, in_channels_1, in_channels_2, num_heads=4):
        super().__init__()
        num_channels = in_channels_1 + in_channels_2
        self.ta = TA()
        # branch 1
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = DWConv(num_channels, num_channels, k=7)
        # branch common
        self.msa_common = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.ln_common = nn.LayerNorm(num_channels)
        # branch 2
        self.msa_2 = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.ln_2 = nn.LayerNorm(num_channels)
        # branch fusion
        self.mca_common_1 = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.ln_common_1 = nn.LayerNorm(num_channels)
        self.fc_common_1 = nn.Linear(num_channels, num_channels)
        self.mca_1_common = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.ln_1_common = nn.LayerNorm(num_channels)
        self.fc_1_common = nn.Linear(num_channels, num_channels)
        self.mca_2_common = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.ln_2_common = nn.LayerNorm(num_channels)
        self.fc_2_common = nn.Linear(num_channels, num_channels)
        self.mca_common_2 = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.ln_common_2 = nn.LayerNorm(num_channels)
        self.fc_common_2 = nn.Linear(num_channels, num_channels)
        # output
        self.conv_out = Conv(num_channels * 3, num_channels, k=1)
        self.bn_out = nn.BatchNorm2d(num_channels)

    def forward(self, x1, x2):
        # x1: [B, C1, H, W]
        # x2: [B, C2, H, W]
        C1 = x1.shape[1]
        C2 = x2.shape[1]
        x = torch.concat((x1, x2), dim=1)  # [B, C1 + C2, H, W]
        B, C, H, W = x.shape
        x = self.ta(x)
        # branch 1
        y_1 = x + self.bn1(self.conv1(x))  # [B, C1 + C2, H, W]
        y_1 = y_1.flatten(2).permute(2, 0, 1)  # [H * W, B, C1 + C2]
        # branch common
        y_common = F.interpolate(x, size=(H//8, W//8))  # [B, C1 + C2, H/8, W/8]
        y_common = y_common.flatten(2).permute(2, 0, 1) # [H * W, B, C1 + C2]
        y_common = y_common + self.ln_common(self.msa_common(y_common, y_common, y_common)[0])
        # branch 2
        y2 = F.interpolate(x, (H//2, W//2)) # [B, C1 + C2, H/2, W/2]
        y2 = y2.flatten(2).permute(2, 0, 1)
        y2 = y2 + self.ln_2(self.msa_2(y2, y2, y2)[0])
        # branch fusion
        y_1 = y_1 + self.fc_common_1(self.ln_common_1(self.mca_common_1(y_1, y_common, y_common)[0]))
        y_common = y_common + self.fc_1_common(self.ln_1_common(self.mca_1_common(y_common, y_1, y_1)[0])) + self.fc_2_common(self.ln_2_common(self.mca_2_common(y_common, y2, y2)[0]))
        y_2 = y2 + self.fc_common_2(self.ln_common_2(self.mca_common_2(y2, y_common, y_common)[0]))
        y_1 = y_1.permute(1, 2, 0).reshape(B, C, H, W)
        y_common = F.interpolate(y_common.permute(1, 2, 0).reshape(B, C, H//8, W//8), size=(H, W))
        y_2 = F.interpolate(y_2.permute(1, 2, 0).reshape(B, C, H//2, W//2), size=(H, W))
        y = torch.concat((y_1, y_common, y_2), dim=1)  # [B, (C1 + C2)*3, H, W]
        y = self.bn_out(self.conv_out(y))
        return y[:,:C1,:,:], y[:,C1:,:,:]  # split into two branches: [B, C1, H, W], [B, C2, H, W]