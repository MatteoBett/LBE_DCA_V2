from typing import Dict
import os
import time

import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import redseq.utils as utils
from redseq.loader import DatasetDCA
import redseq.gapbias as gapbias
from redseq.stats import contribution_profile

def get_freq_single_point(data, weights=None):
    if weights is not None:
        return (data * weights[:, None, None]).sum(dim=0)
    else:
        return data.mean(dim=0)


def get_freq_two_points(data, weights=None):
    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)
    if weights is not None:
        we_data_oh = data_oh * weights[:, None]
    else:
        we_data_oh = data_oh * 1./M
    fij = we_data_oh.T @ data_oh  # Compute weighted sum
    return fij.reshape(L, q, L, q)


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



def init_chains(q, num_chains, L, fi : torch.Tensor | None = None, device :str = "cpu", null=False):
    if not null:
        chains = torch.randint(low=0, high=q, size=(num_chains, L))
    else:
        chains = torch.multinomial(fi, num_samples=num_chains, replacement=True).to(device=device).T
    return utils.get_one_hot(chains)


def gibbs(chains, params : Dict[str, torch.Tensor], beta : int, nb_steps : int, fixed_gaps : torch.Tensor | None = None):
    """Performs a Gibbs sweep over the chains."""
    N, L, q = chains.shape
    for steps in range(nb_steps):
        residue_idxs = torch.randperm(L)
        for i in residue_idxs:
            if fixed_gaps is not None:
                if i in fixed_gaps:
                    chains[:, i,:] = utils.get_one_hot(data=torch.tensor([0])).squeeze(1)
                else:
                    couplings_residue = params["couplings"][i].reshape(q, L * q)
                    fields_residue = params["fields"][i].unsqueeze(0)
                    fields_residue[:, 0] += params['gaps_bias'][i]

                    logit_residue = (fields_residue + chains.reshape(N, L * q) @ couplings_residue.T)[:, 1:]
                    chains[:, i, :] = utils.get_one_hot(torch.multinomial(torch.softmax(logit_residue, dim=-1), 1) + 1, num_classes=q).squeeze(1)
            else:
                couplings_residue = params["couplings"][i].reshape(q, L * q)
                fields_residue = params["fields"][i].unsqueeze(0)
                fields_residue[:, 0] += params['gaps_bias'][i]

                logit_residue = fields_residue + chains.reshape(N, L * q) @ couplings_residue.T
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

def update_params(
    fi: torch.Tensor,
    fij: torch.Tensor,
    pi: torch.Tensor,
    pij: torch.Tensor,
    params: Dict[str, torch.Tensor],
    lr: float,
) -> Dict[str, torch.Tensor]:
    """Updates the parameters of the model.
    """
    
    # Compute the gradient
    grad = compute_gradient_centred(fi=fi, fij=fij, pi=pi, pij=pij)
    
    # Update parameters
    with torch.no_grad():
        for key in params:
            if key == "fields" or key == "couplings":
                params[key] += lr * grad[key]
    
    return params


def fit_model(dataset : DatasetDCA, max_step, min_pearson, lr=0.05, N=10000, nb_gibbs=10, beta=1):
    M, L, q = dataset.mat.shape

    f_single, f_double = get_freq_single_point(dataset.mat), get_freq_two_points(dataset.mat)
    dataset.params['fields'] = torch.log(f_single)
    dataset.params["fields"] = torch.where(torch.isinf(dataset.params["fields"]), torch.tensor(-1e16, device=dataset.params["fields"].device), dataset.params["fields"])

    chains = init_chains(q, N, L)
    p_single, p_double = get_freq_single_point(chains), get_freq_two_points(chains)
     
    halt_condition = lambda s, p: s >= max_step or p >= min_pearson

    pearson = utils.two_points_correlation(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
    step = 0

    while not halt_condition(step, pearson):
        dataset.params = update_params(fi=f_single, 
                                       fij=f_double, 
                                       pi=p_single, 
                                       pij=p_double, 
                                       params=dataset.params, 
                                       lr=lr)
        chains = gibbs(chains, dataset.params, beta, nb_gibbs)
        p_single, p_double = get_freq_single_point(chains), get_freq_two_points(chains)
        pearson = utils.two_points_correlation(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
        step += 1
        
        print(step, pearson)
    

def sample_model(halt_condition, 
                 pearson : float,
                 gap_fraction : float,
                 chains: torch.Tensor,
                 dataset : DatasetDCA,
                 max_sweeps : int,
                 f_single : torch.Tensor,
                 f_double : torch.Tensor,
                 bias_flag : bool | None = None,
                 target_gap_distribution : torch.Tensor | None = None, 
                 constant_bias : bool | None = None,
                 fixed_gaps : bool | None = None):
    
    N, L, q = dataset.mat.shape
    fix_indexes = None
    if fixed_gaps:
        n_fix = int(L*target_gap_distribution.mean().item())
        fix_indexes = torch.topk(f_single[:,0], k=n_fix).indices   
        print(len(fix_indexes)/193)   

    step = 0
    while not halt_condition(s=step, p=pearson, gf=gap_fraction):
        samples = gibbs(
            chains=chains,
            params=dataset.params,
            beta=1.0,
            nb_steps=max_sweeps,
            fixed_gaps=fix_indexes
        )
        p_single, p_double = get_freq_single_point(chains), get_freq_two_points(chains)
        pearson = utils.two_points_correlation(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
        
        if bias_flag:
            gapbias.compute_gap_gradient(
                target_dist=target_gap_distribution,
                dist_sample=p_single[:,0],
                params=dataset.params,
                constant_bias=constant_bias
            )
        step+=1
        print("sampling", step, pearson, p_single[:, 0].mean().item())
    return samples

def sample_trained(
        dataset : DatasetDCA,
        max_sweeps : int,
        num_gen : int,
        sample_it : int,
        min_pearson : float,
        gap_fraction : float,
        bias_flag : bool,
        indel : bool,
        seq_fraction : int,
        constant_bias : bool,
        fixed_gaps : bool,
        bin : int | None = None
    ):
    """ Sample a trained DCA model using Gibbs sampling """
    N, L, q = dataset.mat.shape
    f_single, f_double = get_freq_single_point(data=dataset.mat), get_freq_two_points(data=dataset.mat)
    chains = init_chains(q, num_chains=num_gen, L=L)
    p_single, p_double = get_freq_single_point(chains), get_freq_two_points(chains)
    #dataset.params["fields"] = torch.where(torch.isinf(dataset.params["fields"]), torch.tensor(-1e16, device=dataset.params["fields"].device), dataset.params["fields"])

    halt_condition = lambda s, p, gf: (s >= sample_it) or (p >= min_pearson*(1-0.0033) and (gf < gap_fraction))
    pearson = utils.two_points_correlation(fi=f_single, fij=f_double, pi=p_single, pij=p_double)

    fi_target_gap_distribution = f_single[:, 0].cpu()
    target_gap_distribution = gapbias.get_target_gap_distribution(frac_target=gap_fraction, 
                                                            data_distrib=fi_target_gap_distribution, 
                                                            indel=indel,
                                                            constant_bias=constant_bias).clamp(max=1)
    print("target average gap frequency", target_gap_distribution.mean().item())

    samples = sample_model(halt_condition=halt_condition,
                           pearson=pearson,
                           gap_fraction=gap_fraction,
                           chains=chains,
                           dataset=dataset,
                           max_sweeps=max_sweeps,
                           f_single=f_single,
                           f_double=f_double)

    null_model = init_chains(q, num_chains=num_gen, L=L, fi=dataset.params["fields"].exp(), null=True)
    energies = compute_energy_confs(x=samples, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
    null_energies = compute_energy_confs(x=null_model, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
    
    null_file = os.path.join(os.path.dirname(dataset.chains_unbiased_file), f"null_model{seq_fraction}_{bin}.fasta") if bin is not None else os.path.join(os.path.dirname(dataset.chains_biased_file), f"null_model{seq_fraction}.fasta")
    utils.save_samples(chains=null_model, chains_file=null_file, energies=null_energies)
    utils.save_samples(chains=samples, chains_file=dataset.chains_unbiased_file, energies=energies)
    utils.save_params(infile=dataset.params_file_unbiased, params=dataset.params)


    if bias_flag or constant_bias:
        chains = init_chains(q, num_chains=num_gen, L=L)
        p_single, p_double = get_freq_single_point(chains), get_freq_two_points(chains)
        pearson = utils.two_points_correlation(fi=f_single, fij=f_double, pi=p_single, pij=p_double)
        samples = sample_model(halt_condition=halt_condition,
                               pearson=pearson,
                                gap_fraction=gap_fraction,
                                chains=chains,
                                dataset=dataset,
                                max_sweeps=max_sweeps,
                                f_single=f_single,
                                f_double=f_double,
                                bias_flag=bias_flag,
                                target_gap_distribution=target_gap_distribution,
                                constant_bias=constant_bias,
                                fixed_gaps=fixed_gaps
                                )

        null_model = init_chains(q, num_chains=num_gen, L=L, fi=dataset.params["fields"].exp(), null=True)
        energies = compute_energy_confs(x=samples, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
        null_energies = compute_energy_confs(x=null_model, h_parms=dataset.params["fields"], j_parms=dataset.params["couplings"])
        
        null_file = os.path.join(os.path.dirname(dataset.chains_biased_file), f"null_model{seq_fraction}_{bin}.fasta") if bin is not None else os.path.join(os.path.dirname(dataset.chains_biased_file), f"null_model{seq_fraction}.fasta")
        utils.save_samples(chains=null_model, chains_file=null_file, energies=null_energies)
        utils.save_samples(chains=samples, chains_file=dataset.chains_biased_file, energies=energies)
        utils.save_params(infile=dataset.params_file_biased, params=dataset.params)

