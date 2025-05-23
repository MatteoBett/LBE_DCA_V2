import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import os
from sklearn.metrics import average_precision_score
from model import RNA_Conv1D, RNADataset, train, prepare_train_val, RNA_MLP, RNA_Attention, RNA_reg
from utils import read_fasta, hamming_distance, count_params, read_dca_from_text, msa_to_oh
import matplotlib.pyplot as plt
from tqdm import tqdm

ref = "GUGCCUUGCGCCGGGAAACCACGCAAGGGAUGGUGUCAAAUUCGGCGAAACCUAAGCGCCCGCCCGGGCGUAUGGCAACGCCGAGCCAAGCUUCGGCGCCUGCGCCGAUGAAGGUGUAGAGACUAGACGGCACCCACCUAAGGCAAACGCUAUGGUGAAGGCAUAGUCCAGGGAGUGGCGAAAGUCACACAAACCGG"
struct = "(((((((..((....)).)))))))...((((((....((((((...((...((((((....))))))..))...))))))(((...(.((((((....)))))).)..)))...[.[[[[[...))))))((((...(((....)))..))))......]]]]]]..((.(((((....))))).....))."
base_dir = r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/sampled_seq/Azoarcus'

def compute_energy_confs(x : torch.Tensor, h_parms: torch.Tensor, j_parms: torch.Tensor) -> torch.Tensor:
    M, L, q = x.shape

    # Flatten along the last two dimensions (L*q) for batch processing
    x_oh = x.reshape(M, L * q)
    bias_oh = h_parms.view(-1)  # Flatten bias
    couplings_oh = j_parms.reshape(L * q, L * q)

    # Compute energy contributions
    field = - torch.matmul(x_oh, bias_oh)  # Shape (M,)
    couplings = - 0.5 * torch.einsum('mi,ij,mj->m', x_oh, couplings_oh, x_oh)  # Shape (M,)

    return field + couplings

def runway(tested_seq : str, path_tested_seq : str, fully_connected_dca : str, output : str, sampled_seq : str):
    os.makedirs(output, exist_ok=True)
    msa = read_fasta(tested_seq)
    msa_scan = read_fasta(sampled_seq)
    data_all = pd.read_csv(path_tested_seq, sep=" ", header=None)
    data_all.columns = ["type", "name", "fref", "fsel", "activity"]

    data_all.loc[:, "pos"] = data_all.activity > -2.76
    data_all.loc[:, "nb_mut"] = [hamming_distance(msa[name], ref) for name in data_all.name]
    seq_len = 193
    loss_fn = nn.BCELoss()

    data_train = data_all[data_all.type == "3D"]
    seq_train = [msa[n][3:-1] for n in data_train.name]
    train_loader, val_loader = prepare_train_val(seq_train, data_train.pos.tolist())

    data_test = data_all[data_all.type == "dca_f"]
    seq_all = [msa[n][3:-1] for n in data_test.name]
    test_set = RNADataset(seq_all, data_test.pos.tolist(), pad_token='-')

    seqs_scan = [msa_scan[n] for n in msa_scan.keys()]
    test_scan = RNADataset(sequences=seqs_scan, targets=[-1]*len(seqs_scan), pad_token='-')
    results = {n: [] for n in ["REG", "MLP"]}
    preds_test = {n: [] for n in ["REG", "MLP"]}
    results_scan = {n: [] for n in ["REG", "MLP"]}

    # Model setup
    for model_name in ["REG", "MLP"]:
        for run in tqdm(list(range(20)), desc="Epochs"):
            if model_name == "REG":
                model = RNA_reg(seq_len=seq_len)
            else:
                model = RNA_MLP(seq_len=seq_len, hidden_dim=32, num_layers=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Train
            train(model, train_loader, val_loader, optimizer, loss_fn)

            test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
            scantest_loader = torch.utils.data.DataLoader(test_scan, batch_size=32)
            
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, mask, y in test_loader:
                    preds = model(x, mask)
                    all_preds.append(preds)
                    all_labels.append(y)

            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()

            bins = range(0, data_test.nb_mut.max() + 1, 5)
            out_stat = []
            for i in bins:
                mask = (data_test.nb_mut > i) & (data_test.nb_mut <= i+5)
                if mask.sum() == 0:
                    continue
                out_stat.append(average_precision_score(data_test.pos[mask], preds[mask]))

            h_parms, j_parms = read_dca_from_text(fully_connected_dca)
            seq_oh = msa_to_oh([msa[name][3:-1] for name in data_test.name])
            seq_nrj = compute_energy_confs(seq_oh, h_parms, j_parms)
            out_dca, out_stat = [], []
            bin_l = []
            for i in bins:
                mask = (data_test.nb_mut > i) & (data_test.nb_mut <= i+5)
                if mask.sum() == 0:
                    continue
                bin_l += [i]
                out_dca.append(average_precision_score(data_test.pos[mask], seq_nrj.detach().numpy()[mask]))
                out_stat.append(average_precision_score(data_test.pos[mask], preds[mask]))
            results[model_name] += [out_stat]
            preds_test[model_name] = preds

        model.eval()
        scan_preds = []
        with torch.no_grad():
            for x, mask, y in scantest_loader:
                preds = model(x, mask)
                scan_preds.append(preds)
        print(model_name)
        scan_preds = torch.cat(scan_preds).numpy()
        results_scan[model_name] = scan_preds

    col = {"REG": "C0", "MLP": "C1"}
    plt.rcParams.update({
        "font.size": 6,
        "axes.titlesize": 8,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6
    })
    fig, ax = plt.subplots(figsize=(3, 2))
    fig_scan, scax = plt.subplots(2, 2, figsize=(3, 2))
    for n, model_name in enumerate(results):
        for i in range(20):
            if i == 0:
                ax.plot(np.array(bin_l)+2.5, results[model_name][i], label=model_name, c=col[model_name], alpha=0.4)
                scax[n, 0].scatter(list(range(len(results_scan[model_name]))), results_scan[model_name], marker='.', c=col[model_name])
                scax[n, 0].set_title(f"{model_name} sampled data")
                scax[n, 0].vlines(list(range(0, len(results_scan[model_name]), 500)), ymin=0, ymax=1)
                scax[n, 1].scatter(list(range(len(preds_test[model_name]))), preds_test[model_name], marker='.', c=col[model_name])
                scax[n, 1].set_title(f"{model_name} test data")
                
            else:
                ax.plot(np.array(bin_l)+2.5, results[model_name][i], c=col[model_name], alpha=0.4)
                scax[n, 0].scatter(list(range(len(results_scan[model_name]))),results_scan[model_name], marker='.', c=col[model_name])
                scax[n, 0].set_title(model_name)
                scax[n, 0].vlines(list(range(0, len(results_scan[model_name]), 500)), ymin=0, ymax=1)
                scax[n, 1].scatter(list(range(len(preds_test[model_name]))), preds_test[model_name], marker='.', c=col[model_name])
                scax[n, 1].set_title(f"{model_name} test data")

    fig_scan.savefig(os.path.join(output, "redseq_res.png"), dpi=300)
    ax.plot(np.array(bin_l)+2.5, out_dca, label="DCA", c="black")
    ax.legend(frameon=False)
    ax.set_xlabel("Nb. mutations")
    ax.set_ylabel("Average precision score")
    ax.set_title("3D $\\rightarrow$ DCA T=1")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output, "cross_bench_3D_dca.png"), dpi=300)
    plt.show()

for _dir in os.listdir(base_dir):
    if _dir.startswith("scan"):
        scanpath = os.path.join(base_dir, _dir, f"{_dir}.fasta")
        if os.path.exists(scanpath):
            runway(tested_seq=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/active_seq/all_tested.fasta',
                path_tested_seq=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/active_seq/all_tested.dat',
                fully_connected_dca=r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/data/active_seq/fully_connected.dca',
                output=os.path.join(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/figures/Azoarcus/', _dir),
                sampled_seq=scanpath)
