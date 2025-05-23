import numpy as np
seq= list('CCUUGCGCCGGGAAACCACGCAAGGGAUGGUGUCAAAUUCGGCGAAACCUAAGCGCCCGCCCGGGCGUAUGGCAACGCCGAGCCAAGCUUCGGCGCCUGCGCCGAUGAAGGUGUAGAGACUAGACGGCACCCACCUAAGGCAAACGCUAUGGUGAAGGCAUAGUCCAGGGAGUGGCGAAAGUCACACAAACCG')
idxes = [np.array([np.random.randint(low=0, high=len(seq), size=count) for _ in range(500)]) for count in range(5, 100, 5)]
with open(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sampled_seq/Azoarcus/rdm_azo.fasta', 'w') as rdn:
    for i, idx in enumerate(idxes):
        for k, liste in enumerate(idx):
            new = seq.copy()
            for j in liste:
                new[j] = '-'
            rdn.write(f">seq_bin_{i}_num_{k}\n{"".join(new)}\n")

