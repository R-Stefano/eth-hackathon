import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
sys.path.append('SenCID')
from SenCID.DataPro import ScaleData
from SenCID.Pred import Pred, Recommend

@dataclass
class CellState:
    """Represents a cell's expression state before/after intervention."""
    cell_id: str
    expression_norm: pd.DataFrame
    iteration: int = 0

    def __str__(self):
        return f"CellState(cell_id={self.cell_id}, iteration={self.iteration}, genes={self.expression_norm.shape})"

_counts_cache = None

def reset() -> CellState:
    """Reset the environment to the initial state."""
    global _counts_cache
    
    # Cache the data to avoid reloading every episode
    if _counts_cache is None:
        _counts_cache = pd.read_csv('SenCID/SenCID/demo/demo/origin_matrix_GSE94980.txt', sep='\t', index_col=0).T
        print(f"Loaded: {_counts_cache.shape[0]} cells x {_counts_cache.shape[1]} genes from GSE94980.txt")
    
    # Hardcoded cell for reproducibility
    cell_id = _counts_cache.index[0]  # Always use first cell
    expression = _counts_cache.loc[cell_id]
    adata = sc.AnnData(pd.DataFrame([expression]))
    adata.obs_names = [cell_id]
    
    data_scaled, _ = ScaleData(adata=adata, denoising=False, threads=1, savetmp=False)
    cell_state = CellState(cell_id=cell_id, expression_norm=data_scaled)
    return cell_state


def perform_step(cell_state: CellState, action: Tuple[str, str]) -> Tuple[CellState, Dict, bool]:
    """Perform a step in the environment.
    
    1. Apply intervention to gene expression
    2. Run SenCID prediction on the new state
    
    Args:
        cell_state: Current cell state
        action: Tuple of (gene_name, "ON" or "OFF")
    
    Returns:
        Tuple of (new_cell_state, sid_scores_dict)
    """
    gene, action_type = action

    data_scaled = cell_state.expression_norm.copy()
    print("data_scaled", data_scaled.loc[gene])
    
    # Apply intervention: ON = max, OFF = min
    if gene not in data_scaled.index:
        raise ValueError(f"Gene {gene} not found in expression data")
        
    if action_type == "ON":
        data_scaled.loc[gene] = data_scaled.values.max()
    else:  # OFF
        data_scaled.loc[gene] = data_scaled.values.min()
    print(f"Set {gene} to {data_scaled.loc[gene].values[0]:.3f} ({action_type})")
    
    sidnums = [1, 2, 3, 4, 5, 6]
    pred_list = [Pred(data_scaled, sidnum, binarize=True) for sidnum in sidnums]
    pred_dict = dict(zip(['SID' + str(n) for n in sidnums], pred_list))

    cell_state_next = CellState(cell_id=cell_state.cell_id, expression_norm=data_scaled)
    print("data_scaled", cell_state_next.expression_norm.loc[gene])
    
    return cell_state_next, pred_dict, False


def get_reward(sid_scores: Dict, alpha_correct: float = 1.0, alpha_uncertainty: float = 0.1, eps: float = 1e-9) -> torch.Tensor:
    """
    Goal: Reward one class and penalize the other 5 classes equally
    Note: Move the reward to the device 
    """
    target_idx = 1
    # Convert scores to probabilities
    # sid_scores = torch.tensor([sid_scores[i] for i in range(6)])
    sid_scores = torch.tensor([sid_scores[f'SID{i}']['SID_Score'].values[0] for i in range(1, 7)])
    probs = torch.softmax(sid_scores, dim = 0)
    prob_target = probs[target_idx]

    entropy = -torch.sum(probs * torch.log(probs + eps))
    # Reward the correct class and penalize uncertainty
    reward = alpha_correct * prob_target - alpha_uncertainty * entropy

    return reward
