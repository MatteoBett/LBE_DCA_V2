import os

import torch
import numpy as np

import redseq.loader as dcaload

def sim_nat_vs_gen(path_nat : str, path_gen :str):
    mat_nat = dcaload.DatasetDCA(path_data=path_nat).mat
    mat_gen = dcaload.DatasetDCA(path_data=path_gen).mat

    n, l, q = mat_nat.shape
    N, L, Q = mat_gen.shape

    mat_nat = mat_nat.reshape(n, l*q)
    mat_gen = mat_gen.reshape(N, L*Q)

    similarity = torch.einsum("nd, Nd -> n", mat_nat, mat_gen)/(N*L*Q)
    print(similarity)