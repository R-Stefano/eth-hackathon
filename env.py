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

def reset(cell_index: int = None, max_cells_number: int = None) -> CellState:
    """Reset the environment to the initial state.
    
    Args:
        cell_index: Integer index of cell to use (default: 0 = first cell)
    """
    global _counts_cache
    
    # Cache the data to avoid reloading every episode
    if _counts_cache is None:
        _counts_cache = pd.read_csv('SenCID/SenCID/demo/demo/origin_matrix_GSE94980.txt', sep='\t', index_col=0).T
        print(f"Loaded: {_counts_cache.shape[0]} cells x {_counts_cache.shape[1]} genes from GSE94980.txt")
    
    if cell_index is None and max_cells_number is None:
        cell_index = np.random.randint(0, len(_counts_cache))
    elif cell_index is None and max_cells_number is not None:
        cell_index = np.random.randint(0, max_cells_number)
    else:
        cell_index = cell_index
    print(f"Using cell index: {cell_index}")
    # Use integer index to get cell
    cell_id = _counts_cache.index[cell_index]
    expression = _counts_cache.loc[cell_id]
    adata = sc.AnnData(pd.DataFrame([expression]))
    adata.obs_names = [cell_id]
    
    data_scaled, _ = ScaleData(adata=adata, denoising=False, threads=1, savetmp=False)
    cell_state = CellState(cell_id=cell_id, expression_norm=data_scaled)
    return cell_state

from configs import VERBOSE, TARGET_SID
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
    
    # Apply intervention: ON = max, OFF = min
    if gene not in data_scaled.index:
        raise ValueError(f"Gene {gene} not found in expression data")
        
    if action_type == "ON":
        data_scaled.loc[gene] = data_scaled.values.max()
    else:  # OFF
        data_scaled.loc[gene] = data_scaled.values.min()
    if VERBOSE:
        print(f"Set {gene} to {data_scaled.loc[gene].values[0]:.3f} ({action_type})")
    
    sidnums = [1, 2, 3, 4, 5, 6]
    pred_list = [Pred(data_scaled, sidnum, binarize=True) for sidnum in sidnums]
    pred_dict = dict(zip(['SID' + str(n) for n in sidnums], pred_list))

    cell_state_next = CellState(cell_id=cell_state.cell_id, expression_norm=data_scaled)
    
    # Check if target SID is now the winner
    terminated = check_termination(pred_dict)
    
    return cell_state_next, pred_dict, terminated


def get_winning_sid(sid_scores: Dict) -> Tuple[str, int]:
    """Get the SID with highest score."""
    scores = {f'SID{i}': sid_scores[f'SID{i}']['SID_Score'].values[0] for i in range(1, 7)}
    winning_sid = max(scores, key=scores.get)
    winning_idx = int(winning_sid.split('SID')[1])
    return winning_sid, winning_idx


def check_termination(sid_scores: Dict) -> bool:
    """Terminate if target SID has highest score."""
    winning_sid, _ = get_winning_sid(sid_scores)
    return winning_sid == TARGET_SID


def get_reward(sid_scores: Dict, alpha_correct: float = 1.0, alpha_uncertainty: float = 0.1, eps: float = 1e-9) -> torch.Tensor:
    """
    Goal: Reward one class and penalize the other 5 classes equally
    """
    # Convert scores to tensor
    scores_tensor = torch.tensor([sid_scores[f'SID{i}']['SID_Score'].values[0] for i in range(1, 7)])
    target_idx = int(TARGET_SID.split('SID')[1]) - 1
    probs = torch.softmax(scores_tensor, dim=0)
    prob_target = probs[target_idx]

    if check_termination(sid_scores):
        return torch.tensor(1.0)

    entropy = -torch.sum(probs * torch.log(probs + eps))
    # Reward the correct class and penalize uncertainty
    reward = alpha_correct * prob_target - alpha_uncertainty * entropy

    return reward.detach()  # Detach: rewards are scalar constants in REINFORCE
