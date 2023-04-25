from __future__ import print_function, division
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        # self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.op = nn.Conv3d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        # N, C, W, H = l.size()
        N, C, T, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xTxWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,T,-1), dim=3).view(N,1,T,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            # g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
            g = g.view(N, C, T, -1).sum(dim=3).view(N, C, T, 1, 1)  # batch_sizexC
        else:
            # g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)
            g = F.adaptive_avg_pool3d(g, (T,1,1)).view(N, C, T, 1, 1)
        # return c.view(N,1,W,H), g
        return c.view(N, 1, T, W, H), g

class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C)
        return c.view(N,1,W,H), output
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_conv = 2, pool = False, conv_type=None):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_conv = num_conv

        conv3x3x3 = nn.Conv3d
    
        layers = []
        channels = [in_channel] + [out_channel for i in range(num_conv)]
        for i in range(len(channels) - 1):
            if pool:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
            layers.append(
                nn.Conv3d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, padding=1, padding_mode='replicate', bias=True))
            layers.append(nn.BatchNorm3d(num_features=channels[i + 1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        activation = self.op(x)
        return activation

class conv3D_module(nn.Module):
    def __init__(self, in_channel, out_channel, last_layer=False, conv_type=None):
        super(conv3D_module, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        conv3x3x3 = nn.Conv3d

        # self.maxpool = nn.MaxPool3d((1, 2, 2), stride=None, padding=0)
        if last_layer == False:
            self.conv3d1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=(1, 1, 1),
                    padding_mode='replicate',
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(inplace=True),  # activation
            )
        else:
            self.conv3d1 = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=(3, 3, 3),
                    stride=1,
                    padding=(1, 1, 1),
                    padding_mode='replicate',
                ),
                nn.BatchNorm3d(out_channel),
                nn.Tanh(),  # activation
            )

    def forward(self, x):
        activation = self.conv3d1(x)
        return activation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, last_block = False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = conv3D_module(in_channels, out_channels, last_layer = False)
        self.conv2 = conv3D_module(out_channels, out_channels, last_layer = last_block)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = residual + out
        out = self.relu(out)
        return out