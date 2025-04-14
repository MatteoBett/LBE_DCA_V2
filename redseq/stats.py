import os
from typing import Dict, List

import torch
import numpy as np

import redseq.dca as dca


def compute_Mutual_information(seqs : torch.Tensor, params : Dict[str, torch.Tensor], idx : int, nuc : int):
    mask = (seqs[:, idx, nuc] == 1).nonzero().squeeze(1)
    nucseqs = seqs[mask] #select all sequences where the i-th nucleotide is equal to nuc
    fields = params["fields"].clone()
    couplings = params["couplings"].clone()

    """
    fields[idx, :] = torch.tensor([-0.25]*q)
    fields[idx, nuc] = 1.
    print(couplings[idx, nuc, :, :])
    """

    energies = dca.compute_energy_confs(x=nucseqs, h_parms=fields, j_parms=couplings)
    nucseqs[:, idx, :] = 0
    nucseqs[:, idx, 0] = 1
    energies_env = dca.compute_energy_confs(x=nucseqs, h_parms=fields, j_parms=couplings)
    
    return mask, energies.type(dtype=torch.float32), energies_env.type(dtype=torch.float32)


def contribution_profile(seqs : torch.Tensor, params : Dict[str, torch.Tensor]):
    assert seqs.dim() == 3, "The individual sequences tensor should be of dimension 3!"
    N, L, q = seqs.shape

    proba = torch.zeros((N, L, q), dtype=torch.float32)
    envs = torch.zeros((N, L, q), dtype=torch.float32)
    
    for idx in range(L):
        for nuc in range(q):
            mask, energies, energies_env = compute_Mutual_information(seqs, params, idx, nuc)
            proba[mask, idx, nuc], envs[mask, idx, nuc] = energies, energies_env


    proba = proba.exp()
    envs = envs.exp()

    proba = torch.log2(torch.where(torch.isinf(proba), torch.tensor(0, dtype=torch.float32), proba))
    envs = torch.log2(torch.where(torch.isinf(envs), torch.tensor(0, dtype=torch.float32), envs))
    
    Z1 = torch.logsumexp(proba, dim=[0,2])
    Z2 = torch.logsumexp(envs, dim=[0,2])

    proba = proba - Z1.view(1, L, 1)
    envs = envs - Z2.view(1, L, 1)

    proba = torch.where(torch.isinf(proba), torch.tensor(-1e16, dtype=torch.float32), proba).exp()
    envs = torch.where(torch.isinf(envs), torch.tensor(-1e16, dtype=torch.float32), envs).exp()

    MI = (proba * torch.log2(proba/envs)).nan_to_num(0.0)
    global_profile = MI.sum(dim=0)

    return MI, global_profile
