from __future__ import print_function, division
import os
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock, conv3D_module, ResidualBlock, Decoupled_LDC_T, LDC_T
from cbam import * # Change_cbam
from tool_function import calc_mean_std, norm_sigma, Permutation
from utils import *


class rPPG_estimator(nn.Module):
    def __init__(self, in_ch, seq_length, conv_type=None):
        super(rPPG_estimator, self).__init__()
        self.in_ch = in_ch
        self.ST_module1 = ConvBlock(in_channel = in_ch, out_channel = in_ch*2, conv_type=conv_type)
        self.ST_module2 = ConvBlock(in_channel = in_ch*2, out_channel = in_ch*2, conv_type=conv_type)
        self.ST_module3 = ConvBlock(in_channel = in_ch*2, out_channel = in_ch*2, conv_type=conv_type)
        self.spatialGlobalAvgpool = nn.AdaptiveAvgPool3d((seq_length, 1, 1))
        self.attn1 = MyCBAM_v3( in_ch*2, in_ch*2 ) # 前一個16是input feature的channel數量，後者是reduction ratio
        self.attn2 = MyCBAM_v3( in_ch*2, in_ch*2 )
        self.attn3 = MyCBAM_v3( in_ch*2, in_ch*2 )
        self.conv2d2 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_ch*2,
                out_channels=1,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=(0, 0, 0),
            )
        )

    def forward(self, rPPG_feat):
        
        # Step 2: Basic model
        feat = self.ST_module1(rPPG_feat)
        feat = self.attn1(feat)
        feat = F.max_pool3d(feat, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        feat = self.ST_module2(feat)
        feat = self.attn2(feat)
        feat = F.max_pool3d(feat, kernel_size=(1,2,2), stride=(1,2,2), padding=0)
        feat = self.ST_module3(feat)
        feat = self.attn3(feat)
        g = self.spatialGlobalAvgpool(feat)
        out = self.conv2d2(g)
        
        return out
    
class Encoder(nn.Module):
    def __init__(self, medium_channels, task, conv_type=None):
        super(Encoder, self).__init__()
        self.task = task # removal or embedd
        self.medium_channels = medium_channels

        conv3x3x3 = nn.Conv3d

        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=3, # RGB
                out_channels=medium_channels,
                kernel_size=(5, 5, 5), 
                stride=1,
                padding=(2, 2, 2),
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(medium_channels),
            nn.ReLU(),
        )

    def forward(self, face):
        # Step 2: Basic model
        feat = self.conv3d(face)

        
        return feat
    
class Separator(nn.Module):
    def __init__(self, in_ch, out_ch, task, conv_type=None):
        super(Separator, self).__init__()
        self.task = task 
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.res1 = ResidualBlock(out_ch, out_ch, stride=1, downsample=None, last_block=False)
        self.res2 = ResidualBlock(out_ch, out_ch, stride=1, downsample=None, last_block=False)
        self.conv_final = conv3D_module(out_ch, out_ch, last_layer=False, conv_type=conv_type)


    def forward(self, face):
        # Step 2: Basic model

        feat = self.res1(face)
        feat = self.res2(feat)
        feat = self.conv_final(feat)
        
        return feat
    
class Classifier(nn.Module):
    def __init__(self, in_ch, class_num, seq_length, task):
        super(Classifier, self).__init__()
        self.task = task 
        self.in_ch = in_ch
        self.class_num = class_num
        self.maxpool_conv1 = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(
                in_channels=in_ch,
                out_channels=4,
                kernel_size=5, 
                stride=1,
                padding=(2, 2, 2),
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.maxpool_conv2 = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(
                in_channels=4,
                out_channels=2,
                kernel_size=5, 
                stride=1,
                padding=(2, 2, 2),
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(2),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            torch.nn.Linear(2*60*20*20, 512), # TODO: C/4*T*H/4*W/4 -> 512 
            nn.ReLU(),
            torch.nn.Linear(512, class_num), # TODO 
            # nn.Softmax(dim=1),
        )


    def forward(self, feat):
        # print(f"feat.size(): {feat.size()}")
        feat = self.maxpool_conv1(feat)
        # print(f"(maxpool_conv1) feat.size(): {feat.size()}")
        feat = self.maxpool_conv2(feat)
        # print(f"(maxpool_conv2) feat.size(): {feat.size()}")
        y = self.classifier(feat)
        # print(f"y.size(): {y.size()}")
        # print(f"y: {y}")
        
        return y    

class Decoder_video(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder_video, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=in_ch, # cat(rPPG, id, domain)
                out_channels=out_ch,
                kernel_size=(5, 5, 5), 
                stride=1,
                padding=(2, 2, 2),
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(out_ch, out_ch, stride=1, downsample=None, last_block=False)
        self.res2 = ResidualBlock(out_ch, out_ch, stride=1, downsample=None, last_block=False)
        self.decode_video = nn.Sequential(
            nn.Conv3d(
                in_channels=out_ch, 
                out_channels=3,
                kernel_size=(5, 5, 5), 
                stride=1,
                padding=(2, 2, 2),
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(3),
            nn.Tanh(),
        )


    def forward(self, rPPG_feat, id_feat, domain_feat):
        input_feat = torch.cat((rPPG_feat, id_feat, domain_feat), 1)
        # Step 2: Basic model
        feat = self.conv3d(input_feat)
        feat = self.res1(feat)
        feat = self.res2(feat)
        video = self.decode_video(feat)
        
        return feat, video
    


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)



class Difficult_Transform(nn.Module):
    def __init__(self, shape):
        super(Difficult_Transform, self).__init__()
        self.alpha = nn.Parameter(torch.randn(shape), requires_grad=True)
        self.alpha.data.normal_(0.,1.)
        self.beta = nn.Parameter(torch.randn(shape), requires_grad=True)
        self.beta.data.normal_(0.,1.)

    def forward(self, feat):
        # Step 2: Basic model
        feat_mean, feat_std = calc_mean_std(feat)
        feat = (feat - feat_mean.expand(feat.size())) / feat_std.expand(feat.size())
        #return norm_sigma(self.alpha)*feat + norm_sigma(self.beta)
        return feat*self.alpha.expand(feat.size()) + self.beta.expand(feat.size())
    


class Project_Head(nn.Module):
    def __init__(self,in_ch, seq_length):
        super(Project_Head, self).__init__()
        self.maxpool_conv1 = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(
                in_channels=in_ch,
                out_channels=4,
                kernel_size=5, 
                stride=1,
                padding=(2, 2, 2),
                padding_mode='replicate',
            ),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*seq_length*40*40, 128)
        )

    def forward(self, feat):
        z = self.project(self.maxpool_conv1(feat))
        z = F.normalize(z)
        return z

        