from typing import Dict

import torch

def get_dist(frac_target : float, 
            data_distrib : torch.Tensor, 
            distrib : torch.Tensor,):
    
    passed = []
    rest = 0
    for index, val in enumerate(data_distrib):
        target_val = val*(frac_target/data_distrib.mean())
        if target_val > 1 :
            rest += target_val -1
            distrib[index] = 1
            passed.append(index)
        else:
            distrib[index] = target_val

    val = rest/(len(distrib)-len(passed))
    distrib = torch.tensor([distrib[i] + val if distrib[i] + val <= 1 else 1 for i in range(len(distrib))],
                          device=distrib.device, dtype=distrib.dtype)

    return distrib

def get_target_gap_distribution(frac_target : float, 
                                f_single : torch.Tensor, 
                                indel : bool = False,
                                entropindel : bool = False,
                                constant_bias : bool = False) -> torch.Tensor:
    """ Determines the target frequency distribution of gaps given an overall mean """
    
    fi_target_gap_distribution = f_single[:, 0].cpu()
    if indel is True and constant_bias is True:
        raise ValueError("Cannot have both indel and constant bias")
    
    if constant_bias:
        distrib = torch.full((len(fi_target_gap_distribution),), fill_value=frac_target, dtype=torch.float32).cpu()
        return distrib
    
    if entropindel:
        Sm = -(fi_target_gap_distribution.mean()*torch.log2(fi_target_gap_distribution.mean()))
        del_dist, in_dist = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu(), torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()
        S_low = -(fi_target_gap_distribution*torch.log2(fi_target_gap_distribution)) < Sm
        S_high = -(fi_target_gap_distribution*torch.log2(fi_target_gap_distribution)) > Sm

        idx_del, idx_in = S_low.nonzero(), S_high.nonzero()
        prop_del, prop_in = (len(idx_del)/len(fi_target_gap_distribution))*frac_target, (len(idx_in)/len(fi_target_gap_distribution))*frac_target

        data_del = fi_target_gap_distribution.where(-(fi_target_gap_distribution*torch.log2(fi_target_gap_distribution)) >= Sm, 0)
        data_in = fi_target_gap_distribution.where(-(fi_target_gap_distribution*torch.log2(fi_target_gap_distribution)) <= Sm, 0)

        del_dist = get_dist(frac_target=prop_del, data_distrib=data_del, distrib=del_dist)
        in_dist = get_dist(frac_target=prop_in, data_distrib=data_in, distrib=in_dist)
        return del_dist + in_dist
    
    if indel:
        del_dist, in_dist = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu(), torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()
        _del = fi_target_gap_distribution < fi_target_gap_distribution.mean()
        _in = fi_target_gap_distribution > fi_target_gap_distribution.mean()
        
        idx_del, idx_in = _del.nonzero(), _in.nonzero()
        prop_del, prop_in = (len(idx_del)/len(fi_target_gap_distribution))*frac_target, (len(idx_in)/len(fi_target_gap_distribution))*frac_target

        data_del = fi_target_gap_distribution.where(fi_target_gap_distribution >= fi_target_gap_distribution.mean(), 0)
        data_in = fi_target_gap_distribution.where(fi_target_gap_distribution <= fi_target_gap_distribution.mean(), 0)

        del_dist = get_dist(frac_target=prop_del, data_distrib=data_del, distrib=del_dist)
        in_dist = get_dist(frac_target=prop_in, data_distrib=data_in, distrib=in_dist)
        return del_dist + in_dist

    else:
        distrib = torch.zeros((len(fi_target_gap_distribution), 1), dtype=torch.float32).cpu()
        return get_dist(frac_target=frac_target, data_distrib=fi_target_gap_distribution, distrib=distrib)

def compute_gap_gradient(target_dist : torch.Tensor,
                         dist_sample : torch.Tensor,
                         params : Dict[str, torch.Tensor],
                         S : torch.Tensor | None = None,
                         fixed_gaps : bool | None = None,
                         adaptative : bool | None = None,
                         constant_bias : bool | None = None,
                         device : str = 'cpu'
                         ) -> torch.Tensor:
    """
    Computes the gradient of the bias applied to the gaps frequency and adjust it 
    toward a target distribution of gaps corresponding to a mean frequency of gaps in the sequence.
    """ 
    target_dist = target_dist.to(device=device)
    dist_sample = dist_sample.to(device=device)

    if constant_bias or fixed_gaps:
        target_dist = target_dist.mean().item()
        dist_sample = dist_sample.mean().item()

    if adaptative:
        target_dist = target_dist.mean().item()
        dist_sample = dist_sample.mean().item()

        loss = target_dist - dist_sample
        T = params["gaps_lr"]*loss

    else:
        loss = target_dist - dist_sample
        new_bias = params["gaps_lr"] * loss  # positive result
        if target_dist == 0.0:
            new_bias = torch.tensor(-1e6)

    if constant_bias:
        params["gaps_bias"][:, 0] = torch.full(params["gaps_bias"][:, 0].shape, new_bias.item())
        
    elif adaptative:
        params["gaps_bias"][:, 0] = S*T
        #params["fields"][:, 0] = torch.where(torch.isinf(params["fields"][:,0]), torch.tensor(-1e16, device=params["fields"][:,0].device), params["fields"][:,0])

    else:
        params["gaps_bias"][:, 0] = new_bias

    """if params["gaps_lr"] > 10e-5:
        params["gaps_lr"] = params["gaps_lr"]*(1-0.0033)"""

    return loss