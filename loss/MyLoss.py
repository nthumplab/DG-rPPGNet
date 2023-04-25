import torch
import numpy as np
import torch.nn as nn
import torch.fft

class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()
        return

    def forward(self, x, y):
        # for i in range(x.shape[0]):
        vx = x - torch.mean(x, dim = 1, keepdim = True)
        vy = y - torch.mean(y, dim = 1, keepdim = True)
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        cost = 1 - r
        return cost

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
        self.ST_GlobalAvgpool = nn.AdaptiveAvgPool3d((1,1,1))
        return

    def forward(self, rPPG_feat, id_feat, domain_feat):
        # print(f"rPPG_feat.size(): {rPPG_feat.size()}")
        # print(f"id_feat.size(): {id_feat.size()}")
        # print(f"domain_feat.size(): {domain_feat.size()}")
        rPPG_vector = self.ST_GlobalAvgpool(rPPG_feat).squeeze()
        id_vector = self.ST_GlobalAvgpool(id_feat).squeeze()
        domain_vector = self.ST_GlobalAvgpool(domain_feat).squeeze()
        # print(f"rPPG_vector.size(): {rPPG_vector.size()}")
        # print(f"id_vector.size(): {id_vector.size()}")
        # print(f"domain_vector.size(): {domain_vector.size()}")
        loss = 0.0
        for i in range(rPPG_vector.shape[0]):
            inner_R_I = torch.inner(rPPG_vector[i], id_vector[i])
            inner_R_D = torch.inner(rPPG_vector[i], domain_vector[i])
            inner_I_D = torch.inner(id_vector[i], domain_vector[i])
            # print(f"rPPG_vector: {rPPG_vector[i]}")
            # print(f"id_vector: {id_vector[i]}")
            # print(f"domain_vector: {domain_vector[i]}")
            # print(f"inner_R_I:\n{inner_R_I}")
            # print(f"inner_R_D:\n{inner_R_D}")
            # print(f"inner_I_D:\n{inner_I_D}")
            loss += (inner_R_I+inner_R_D+inner_I_D)
        return loss

class Cos_Sim_loss(nn.Module):
    
    def __init__(self):
        super(Cos_Sim_loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output, label):
        return 1 - torch.mean(self.cos(output, label))

class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x
    

class PSD_norm_loss(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(PSD_norm_loss, self).__init__()

        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss(reduction = 'mean')


    def forward(self, x, y):
        
        x_psd = self.norm_psd(x)
        y_psd = self.norm_psd(y)

        return self.distance_func(x_psd, y_psd)
    
class Cos_margin_loss(nn.Module):
    
    def __init__(self, margin_rppg, margin_domain, margin_id):
        super(Cos_margin_loss, self).__init__()
        self.cos = nn.CosineSimilarity()
        self.margin_rppg = margin_rppg
        self.margin_domain = margin_rppg
        self.margin_id = margin_rppg

    def forward(self, fwt_feat, rppg_feat, domain_feat, id_feat, epoch):
        #if(epoch < 50):

        #    return torch.mean(self.cos(fwt_feat, rppg_feat))
        
        #else:
        rppg_similarity_loss = (torch.maximum(torch.mean(self.cos(fwt_feat, rppg_feat)) - self.margin_rppg, torch.tensor(0)))
        domain_similarity_loss = (torch.maximum(torch.mean(self.cos(fwt_feat, domain_feat)) - self.margin_domain, torch.tensor(0)))
        id_similarity_loss = (torch.maximum(torch.mean(self.cos(fwt_feat, id_feat)) - self.margin_id, torch.tensor(0)))

        return rppg_similarity_loss + domain_similarity_loss + id_similarity_loss
    

class Cos_margin_loss2(nn.Module):
    
    def __init__(self, margin_rppg, margin_domain, margin_id):
        super(Cos_margin_loss2, self).__init__()
        self.cos = nn.CosineSimilarity()
        self.margin_rppg = margin_rppg
        self.margin_domain = margin_rppg
        self.margin_id = margin_rppg

    def forward(self, fwt_feat, rppg_feat, domain_feat, id_feat, epoch):
        #if(epoch < 50):

        #    return torch.mean(self.cos(fwt_feat, rppg_feat))
        
        #else:
        rppg_similarity = torch.mean(self.cos(fwt_feat, rppg_feat))
        domain_similarity = torch.mean(self.cos(fwt_feat, domain_feat))
        id_similarity = torch.mean(self.cos(fwt_feat, id_feat))

        domain_similarity_loss = torch.maximum(torch.tensor(0), domain_similarity - rppg_similarity + self.margin_domain)
        id_similarity_loss = torch.maximum(torch.tensor(0), id_similarity - rppg_similarity + self.margin_id)

        return domain_similarity_loss + id_similarity_loss