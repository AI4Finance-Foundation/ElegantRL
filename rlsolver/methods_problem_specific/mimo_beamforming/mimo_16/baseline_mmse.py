import torch as th
import numpy as np


def compute_mmse_beamformer(mat_H, K=4, N=4, P=10, noise_power=1, device=th.device("cuda:0")):
    P = th.diag_embed(th.ones(mat_H.shape[0], 1, device=device).repeat(1, K)).to(th.cfloat)
    eye_N = th.diag_embed((th.zeros(mat_H.shape[0], N, device=device) + noise_power))
    denominator = th.inverse(eye_N + th.bmm(mat_H.conj().transpose(1,2), th.bmm(P / K, mat_H)))
    wslnr_max = th.bmm(denominator, mat_H.conj().transpose(1,2)).transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    mat_W = th.bmm(wslnr_max, th.sqrt(P/ K))
    mat_HW = th.bmm(mat_H, mat_W.transpose(-1, -2))
    S = th.abs(th.diagonal(mat_HW, dim1=-2, dim2=-1))**2
    I = th.sum(th.abs(mat_HW)**2, dim=-1) - th.abs(th.diagonal(mat_HW, dim1=-2, dim2=-1))**2
    N = noise_power
    SINR = S/(I+N)
    return mat_W, th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)

def compute_mmse_beamformer_relay(mat_HTFG, mat_H, mat_F, K=4, N=4, P=10, noise_power=1, device=th.device("cuda:0")):
    mat_power_allocation = th.diag_embed(th.ones((mat_HTFG.shape[0], K), dtype=th.cfloat)).to(device) * P / K
    mat_power_allocation_sqrt = th.diag_embed(th.ones((mat_HTFG.shape[0], K), dtype=th.cfloat)).to(device) * np.sqrt(P / K)
    eye_N = th.diag_embed((th.zeros(mat_HTFG.shape[0], N).to(device) + noise_power + mat_F.flatten(start_dim=1).norm(dim=1, keepdim=True))).to(device)
    denominator = th.inverse(eye_N + th.bmm(mat_HTFG.conj().transpose(1,2),th.bmm(mat_power_allocation, mat_HTFG)))
    wslnr_max = th.bmm(denominator, mat_HTFG.conj().transpose(1,2)).transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    mat_W = th.bmm(mat_power_allocation_sqrt, wslnr_max)

    HTF = th.bmm(mat_H.conj().transpose(-1,-2), mat_F)
    HTFGW = th.bmm(mat_HTFG, mat_W.to(th.cfloat).transpose(-1, -2))
    S = th.abs(th.diagonal(HTFGW, dim1=-2, dim2=-1))**2
    I = th.sum(th.abs(HTFGW)**2, dim=-1) - th.abs(th.diagonal(HTFGW, dim1=-2, dim2=-1))**2
    N = th.norm(HTF, dim=-1)**2 * 1 + noise_power
    SINR = S / (I + N)
    return mat_W, th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
