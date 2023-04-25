import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import double


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
    return feat_mean, feat_std

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1

def Permutation(input_tensor, permu_order):
    if len(permu_order) == 3:
        permu_tensor = torch.cat([
            input_tensor[permu_order[0]].unsqueeze(0),
            input_tensor[permu_order[1]].unsqueeze(0),
            input_tensor[permu_order[2]].unsqueeze(0)])
    elif len(permu_order) == 2:
        permu_tensor = torch.cat([
            input_tensor[permu_order[0]].unsqueeze(0),
            input_tensor[permu_order[1]].unsqueeze(0)])
    else:
        print(f"permu_order Error")
    
    return permu_tensor


def getShuffleIdx(batch_size):
            
    same = torch.arange(batch_size)
    shuffle_idx = torch.randperm(batch_size)
    while(torch.equal(same, shuffle_idx)):
        shuffle_idx = torch.randperm(batch_size)
    
    return shuffle_idx

