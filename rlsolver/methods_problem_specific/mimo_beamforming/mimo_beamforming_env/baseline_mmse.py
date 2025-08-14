import torch
import numpy as np


def compute_mmse_beamformer(mat_H, K=4, N=4, P=10, noise_power=1, device=torch.device("cpu")):
    eye_N = torch.diag_embed((torch.zeros(mat_H.shape[0], N) + noise_power)).to(device)
    denominator = torch.inverse(eye_N + torch.bmm(mat_H.conj().transpose(1,2), torch.bmm(P / K, mat_H)))
    wslnr_max = torch.bmm(denominator, mat_H.conj().transpose(1,2)).transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    mat_W = torch.bmm(wslnr_max, torch.sqrt(P/ K))
    mat_HW = torch.bmm(mat_H, mat_W.transpose(-1, -2))
    S = torch.abs(torch.diagonal(mat_HW, dim1=-2, dim2=-1))**2
    I = torch.sum(torch.abs(mat_HW)**2, dim=-1) - torch.abs(torch.diagonal(mat_HW, dim1=-2, dim2=-1))**2
    N = noise_power
    SINR = S/(I+N)
    return mat_W, torch.log2(1+SINR).sum(dim=-1).unsqueeze(-1)

def compute_mmse_beamformer_relay(mat_HTFG, mat_H, mat_F, K=4, N=4, P=10, noise_power=1, device=torch.device("cuda:0")):
    eye_N = torch.diag_embed((torch.zeros(mat_HTFG.shape[0], N).to(device) + noise_power + mat_F.flatten(start_dim=1).norm(dim=1, keepdim=True))).to(device)
    denominator = torch.inverse(eye_N + torch.bmm(mat_HTFG.conj().transpose(1,2), P / K * mat_HTFG))
    wslnr_max = torch.bmm(denominator, mat_HTFG.conj().transpose(1,2)).transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    mat_W = wslnr_max * np.sqrt(P/ K)
    HTF = torch.bmm(mat_H.conj().transpose(-1,-2), mat_F)
    HTFGW = torch.bmm(mat_HTFG, mat_W.to(torch.cfloat).transpose(-1, -2))
    S = torch.abs(torch.diagonal(HTFGW, dim1=-2, dim2=-1))**2
    I = torch.sum(torch.abs(HTFGW)**2, dim=-1) - torch.abs(torch.diagonal(HTFGW, dim1=-2, dim2=-1))**2
    N = torch.norm(HTF, dim=-1)**2 * 1 + noise_power
    SINR = S / (I + N)
    return mat_W, torch.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
