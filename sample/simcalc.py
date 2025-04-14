import os
from typing import Dict, List

import numpy as np

import sample.loader as loader
import redseq.utils as utils

def truth_table(A : int, B : int, A_ss : int, B_ss : int):
    if A == B and A_ss == B_ss:
        return 2
    elif A == B and A_ss != B_ss:
        return 1
    elif A != B and A_ss == B_ss:
        return 1
    elif A != B and A_ss != B_ss:
        return 0
    
def jaccard_distance(lstA : List[int], lstB: List[int]):
    idxA = 0
    idxB = 0

    intersection = 0
    union = 0

    while idxA < len(lstA) and idxB < len(lstB):
        union += 1
        if lstA[idxA] == lstB[idxB] :
            intersection += 1
            idxA += 1
            idxB += 1
        elif lstA[idxA] < lstB[idxB]:
            idxA += 1
        else:
            idxB += 1

    union += len(lstA) - idxA
    union += len(lstB) - idxB

    return intersection / union


def stream_batches_similarity(batches : Dict[int, List[loader.SeqSlope]]):
    all_dist = {}
    for index, (size, batch) in enumerate(batches.items()):
        utils.progressbar(iteration=index+1, total=len(batches))
        dist = []
        for i in range(0, len(batch)):
            for j in range(i, len(batch)):
                if i != j:
                    dist.append(jaccard_distance(lstA=batch[i].encoded, 
                                                lstB=batch[j].encoded))
        all_dist[size] = dist
    return all_dist

def stream_batches_smaller(batches : Dict[int, List[loader.SeqSlope]]):
    keys = list(batches.keys())
    for index, (size, batch) in enumerate(batches.items()):
        utils.progressbar(iteration=index+1, total=len(batches))
        for key in keys[index+1:len(keys)]:
            
            for i in range(0, len(batch)):
                tmp = []
                for j in range(0, len(batches[key])):
                    dist_ij = jaccard_distance(lstA=batch[i].encoded, 
                                                lstB=batches[key][j].encoded)
                
                    tmp.append(dist_ij)
        
                batches[size][i].targets[key] = np.mean(tmp)
        


        