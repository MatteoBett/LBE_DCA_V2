from typing import Dict, List
import os, time

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import redseq.utils as utils
from redseq.loader import DatasetDCA
import redseq.gapbias as gapbias
from redseq.stats import contribution_profile
import redseq.viz.display as display


def init_chains(q, num_chains, L, fi : torch.Tensor | None = None, device :str = "cpu", null : bool =False):
    if not null:
        chains = torch.randint(low=0, high=q, size=(num_chains, L))
    else:
        chains = torch.multinomial(fi, num_samples=num_chains, replacement=True).to(device=device).T
    return utils.get_one_hot(chains)


def gibbs(chains, params : Dict[str, torch.Tensor], beta : int, nb_steps : int, fixed_gaps : torch.Tensor | None = None):
    """Performs a Gibbs sweep over the chains."""
    N, L, q = chains.shape
    for _ in range(nb_steps):
        residue_idxs = torch.randperm(L)
        for i in residue_idxs:
            if fixed_gaps is not None:
                if i in fixed_gaps:
                    chains[:, i,:] = utils.get_one_hot(data=torch.tensor([0])).squeeze(1)
                else:
                    couplings_residue = params["couplings"][i].reshape(q, L * q)
                    fields_residue = params["fields"][i].unsqueeze(0)
                    fields_residue[:, 0] += params['gaps_bias'][i]

                    logit_residue = ((fields_residue + chains.reshape(N, L * q) @ couplings_residue.T)[:, 1:])*beta
                    chains[:, i, :] = utils.get_one_hot(torch.multinomial(torch.softmax(logit_residue, dim=-1), 1) + 1, num_classes=q).squeeze(1)
            else:

                couplings_residue = params["couplings"][i].reshape(q, L * q)
                fields_residue = params["fields"][i].unsqueeze(0)
                fields_residue[:, 0] += params['gaps_bias'][i]

                logit_residue = (fields_residue + chains.reshape(N, L * q) @ couplings_residue.T)*beta
                chains[:, i, :] = utils.get_one_hot(torch.multinomial(torch.softmax(logit_residue, dim=-1), 1), num_classes=q).squeeze(1)
        
    return chains

@torch.jit.script
def compute_gradient_centred(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the log-likelihood of the model using PyTorch.
    Implements the centred gradient, which empirically improves the convergence of the model.
    """
    grad = {}
    
    C_data = fij - torch.einsum("ij,kl->ijkl", fi, fi)
    C_model = pij - torch.einsum("ij,kl->ijkl", pi, pi)
    grad["couplings"] = C_data - C_model
    grad["fields"] = fi - pi - torch.einsum("iajb,jb->ia", grad["couplings"], fi)
    
    return grad

@torch.jit.script
def compute_grad(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    grad = {}
    grad["couplings"] = (fij-pij)
    grad["fields"] = (fi-pi)

    return grad

def update_params(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    lr: float,
    l1 : float = 0.0
) -> Dict[str, torch.Tensor]:
    """Updates the parameters of the model.
    """
    
    # Compute the gradient
    grad = compute_grad(fi=fi, fij=fij, pi=pi, pij=pij)
    
    # Update parameters
    with torch.no_grad():
        centered_fields = params["fields"] - params["fields"].mean()
        centered_couplings = params["couplings"] - params["couplings"].mean()
        reg_fields = l1*np.sign(centered_fields)
        reg_couplings = (l1**2)*np.sign(centered_couplings)
        params["fields"] += lr * grad["fields"] - reg_fields
        params["couplings"] += lr * grad["couplings"] - reg_couplings

    fields_nonzero = np.nonzero(reg_fields)
    couplings_nonzero = np.nonzero(reg_couplings)


def choose_eval(eval : str):
    if eval == "cross-pearson":
        return utils.two_points_correlation

    if eval == "R2":
        return utils.R2_model
    
    else:
        raise ValueError("Non available evaluation method.")

def thresholded_sign(x, thresh=1e-6):
    sign = np.sign(x)
    sign[np.abs(x) < thresh] = 0
    return sign

def fit_model(dataset : DatasetDCA, max_step, min_eval, lr=0.001, N=10000, nb_gibbs=10, beta=1.0, eval_method="pearson"):
    print(f"Convergence evaluated by {eval_method}")
    eval_method = choose_eval(eval=eval_method)
    M, L, q = dataset.mat.shape

    f_single, f_double = utils.get_freq_single_point(dataset.mat), utils.get_freq_two_points(dataset.mat)
    dataset.params['fields'] = torch.log(f_single)
    dataset.params["fields"] = torch.where(torch.isinf(dataset.params["fields"]), torch.tensor(-1e16, device=dataset.params["fields"].device), dataset.params["fields"])

    #dataset.params["fields"] = torch.zeros((L, q))

    chains = init_chains(q, num_chains=N, L=L, fi=dataset.params["fields"].exp(), null=True)
    p_single, p_double = utils.get_freq_single_point(chains), utils.get_freq_two_points(chains)
     
    halt_condition = lambda s, p: s >= max_step or p >= min_eval

    convergence_eval = eval_method(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
    simple_pearson, _ = pearsonr(f_double.ravel(), p_double.ravel())
    
    step = 0
    l1 = 0.01
    print(step, convergence_eval, simple_pearson)

    nonzero_field_list = []
    nonzero_couplings_list = []
    while not halt_condition(step, convergence_eval):
        grad_i = lr * (p_single - f_single)
        grad_ij = lr * (p_double - f_double)
        centered_fields = dataset.params["fields"] - dataset.params["fields"].mean()
        centered_couplings = dataset.params["couplings"] - dataset.params["couplings"].mean()
        reg_fields = l1*thresholded_sign(centered_fields)
        reg_couplings = (l1**2)*thresholded_sign(centered_couplings)
        fields_nonzero = len(np.nonzero(reg_fields))
        couplings_nonzero = len(np.nonzero(reg_couplings))

        dataset.params["fields"] -= grad_i - reg_fields
        dataset.params["couplings"] -= grad_ij - reg_couplings

        chains = gibbs(chains, dataset.params, beta, nb_gibbs)
        p_single, p_double = utils.get_freq_single_point(chains), utils.get_freq_two_points(chains)

        convergence_eval = eval_method(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
        simple_pearson, _ = pearsonr(f_double.ravel(), p_double.ravel())
        
        step += 1

        nonzero_field_list.append(fields_nonzero)
        nonzero_couplings_list.append(couplings_nonzero)
        print(step, convergence_eval, simple_pearson, fields_nonzero, couplings_nonzero)
        #convergence_eval = simple_pearson
    
    plt.plot(nonzero_field_list, marker='o', c='r', label="Fields nonzero")
    plt.plot(nonzero_couplings_list, marker='o', c='b', label='Couplings nonzero')
    plt.xlabel('Epoch training')
    plt.ylabel("Nonzero count")
    plt.yscale('log')
    plt.savefig(r'/home/mbettiati/LBE_MatteoBettiati/code/vdca/output/figures/Azoarcus/Zerocount.png')

def sample_model(halt_condition, 
                 evaluation : float,
                 gap_fraction : float,
                 chains: torch.Tensor,
                 dataset : DatasetDCA,
                 max_sweeps : int,
                 f_single : torch.Tensor,
                 f_double : torch.Tensor,
                 beta : float, 
                 eval_method,
                 adaptative : bool | None = None,
                 bias_flag : bool | None = None,
                 target_gap_distribution : torch.Tensor | None = None, 
                 constant_bias : bool | None = None,
                 fixed_gaps : bool | None = None):
    
    _, L, _ = dataset.mat.shape
    fix_indexes = None
    S = None
    if fixed_gaps:
        n_fix = int(L*target_gap_distribution.mean().item())
        fix_indexes = torch.topk(f_single[:,0], k=n_fix).indices   
        print(len(fix_indexes)/193)   

    if adaptative:
        fN = f_single[:, :4].sum(dim=1)
        fG = f_single[:, 0]

        lfN = torch.where(torch.isinf(torch.log2(fN)), torch.tensor(0, device=fN.device), fN)
        lfG = torch.where(torch.isinf(torch.log2(fG)), torch.tensor(0, device=fN.device), fG)

        S = (fN * lfN + fG * lfG)
    
    step = 0
    while not halt_condition(s=step, p=evaluation, gf=gap_fraction):
        samples = gibbs(
            chains=chains,
            params=dataset.params,
            beta=beta,
            nb_steps=max_sweeps,
            fixed_gaps=fix_indexes,
        )   
        p_single, p_double = utils.get_freq_single_point(chains), utils.get_freq_two_points(chains)
        evaluation = eval_method(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
        simple_pearson, _ = pearsonr(f_double.ravel(), p_double.ravel())
        if bias_flag:
            gapbias.compute_gap_gradient(
                target_dist=target_gap_distribution,
                dist_sample=p_single[:,0],
                params=dataset.params,
                constant_bias=constant_bias,
                adaptative=adaptative,
                S=S
            )
        step+=1
        print("sampling", step, "cross-pearson", evaluation, "simple pearson", simple_pearson, "gap freq", p_single[:, 0].mean().item())
    return samples

def sample_trained(
        dataset : DatasetDCA,
        max_sweeps : int,
        num_gen : int,
        sample_it : int,
        min_eval : float,
        gap_fraction : float,
        bias_flag : bool,
        indel : bool,
        seq_fraction : int,
        constant_bias : bool,
        fixed_gaps : bool,
        eval_method : str,
        beta : float,
        adaptative : bool,
        plotting : bool,
        family_fig_dir : str,
        interpolation_targets : List[float] | None = None,
        full_interpolation : bool | None = None,
    ):
    """ Sample a trained DCA model using Gibbs sampling """
    N, L, q = dataset.mat.shape
    eval_method = choose_eval(eval=eval_method)
    f_single, f_double = utils.get_freq_single_point(data=dataset.mat), utils.get_freq_two_points(data=dataset.mat)

    chains = init_chains(q, num_chains=num_gen, L=L)
    #chains = dataset.mat[torch.randint(len(dataset.mat),(num_gen,))]
    
    p_single, p_double = utils.get_freq_single_point(chains), utils.get_freq_two_points(chains)
    #dataset.params["fields"] = torch.where(torch.isinf(dataset.params["fields"]), torch.tensor(-1e16, device=dataset.params["fields"].device), dataset.params["fields"])

    halt_condition = lambda s, p, gf: (s >= sample_it) or (p >= min_eval*(1-0.0033) and (gf < gap_fraction))
    evaluation = eval_method(fi=f_single, fij=f_double, pi=p_single, pij=p_double)

    samples = sample_model(halt_condition=halt_condition,
                           evaluation=evaluation,
                           gap_fraction=gap_fraction,
                           chains=chains,
                           dataset=dataset,
                           max_sweeps=max_sweeps,
                           beta=beta,
                           f_single=f_single,
                           f_double=f_double,
                           eval_method=eval_method)

    null_model = init_chains(q, num_chains=num_gen, L=L, fi=dataset.params["fields"].exp(), null=True)
    energies = utils.compute_energy_confs(x=samples, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
    null_energies = utils.compute_energy_confs(x=null_model, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
    
    null_file = os.path.join(os.path.dirname(dataset.chains_unbiased_file), f"null_model0.fasta")
    utils.save_samples(chains=null_model, chains_file=null_file, energies=null_energies)
    utils.save_samples(chains=samples, chains_file=dataset.chains_unbiased_file, energies=energies)
    utils.save_params(infile=dataset.params_file_unbiased, params=dataset.params)


    if bias_flag:
        if full_interpolation:
            target_list = interpolation_targets
        else:
            target_list = [gap_fraction]
        
        for gap_fraction in target_list:
            bin = round(gap_fraction*100)
            target_gap_distribution = gapbias.get_target_gap_distribution(frac_target=gap_fraction, 
                                                                    f_single=f_single, 
                                                                    indel=indel,
                                                                    constant_bias=constant_bias).clamp(max=1)
            
            print("target average gap frequency", target_gap_distribution.mean().item())
            chains = init_chains(q, num_chains=num_gen, L=L)
            p_single, p_double = utils.get_freq_single_point(chains), utils.get_freq_two_points(chains)
            evaluation = eval_method(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
            samples = sample_model(halt_condition=halt_condition,
                                evaluation=evaluation,
                                    gap_fraction=gap_fraction,
                                    chains=chains,
                                    dataset=dataset,
                                    max_sweeps=max_sweeps,
                                    f_single=f_single,
                                    f_double=f_double,
                                    bias_flag=bias_flag,
                                    target_gap_distribution=target_gap_distribution,
                                    constant_bias=constant_bias,
                                    fixed_gaps=fixed_gaps,
                                    eval_method=eval_method,
                                    beta=beta,
                                    adaptative=adaptative
                                    )

            null_model = init_chains(q, num_chains=num_gen, L=L, fi=dataset.params["fields"].exp(), null=True)
            energies = utils.compute_energy_confs(x=samples, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
            null_energies = utils.compute_energy_confs(x=null_model, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
            
            null_file = os.path.join(os.path.dirname(dataset.chains_biased_file), f"null_model{seq_fraction}_{bin}.fasta")
            chains_biased_file = f"{dataset.chains_biased_file.split('.')[0]}_{bin}.fasta" 
            params_file_biased = f"{dataset.params_file_biased.split('.')[0]}_{bin}.json"

            utils.save_samples(chains=null_model, chains_file=null_file, energies=null_energies)
            utils.save_samples(chains=samples, chains_file=chains_biased_file, energies=energies)
            utils.save_params(infile=params_file_biased, params=dataset.params)
           

            if plotting:
                display.homology_vs_gaps(chains_file_ref=dataset.chains_unbiased_file, 
                                        infile_path=dataset.path_data, 
                                        chains_file_bias=chains_biased_file,
                                        indel=indel, 
                                        fig_dir=family_fig_dir,
                                        params_path_unbiased=dataset.params_file_unbiased,
                                        params_path_biased=params_file_biased,
                                        constant=constant_bias,
                                        fixed_gaps=fixed_gaps,
                                        eval_method=eval_method,
                                        bin=bin)    

