import sys
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Tuple, Dict

sys.path.append('SenCID')
from SenCID.api import SenCID
from intervention import CellState


def _apply_intervention(state: CellState, action: Tuple[str, str]) -> CellState:
    """Apply a gene intervention to a cell state.
    
    Args:
        state: Current cell state
        action: Tuple of (gene_name, "ON" or "OFF")
    """
    gene, action_type = action
    new_expression = state.expression.copy()
    
    if gene not in new_expression.index:
        raise ValueError(f"Gene {gene} not found in expression data")
    
    current_val = new_expression[gene]
    
    # ON = increase expression, OFF = decrease expression
    if action_type == "ON":
        new_val = current_val * 2.0  # 2x fold increase
    else:  # OFF
        new_val = current_val * 0.1  # 90% knockdown
    
    new_expression[gene] = max(0, new_val)  # expression >= 0
    
    return CellState(
        cell_id=f"{state.cell_id}_intervened",
        expression=new_expression,
        intervention={
            "gene": gene,
            "action": action_type,
            "original_value": current_val,
            "new_value": new_expression[gene]
        }
    )


def _run_sencid(cell_state: CellState) -> Dict:
    """Run SenCID on a cell state and return SID scores."""
    adata = sc.AnnData(pd.DataFrame([cell_state.expression]))
    adata.obs_names = [cell_state.cell_id]
    pred_dict, rec_sid, _ = SenCID(adata, denoising=False, binarize=True)
    return pred_dict


def perform_step(cell_state: CellState, action: Tuple[str, str]) -> Tuple[CellState, Dict]:
    """Perform a step in the environment.
    
    Returns:
        Tuple of (new_cell_state, sid_scores_dict)
    """
    cell_state_next = _apply_intervention(cell_state, action)
    sid_scores = _run_sencid(cell_state_next)
    return cell_state_next, sid_scores


def get_reward(sid_scores: Dict) -> float:
    """Calculate reward based on cell state and SID scores.
    
    Goal: Push cells toward high immunogenicity (higher MHC-I visibility).
    We reward lower senescence scores (SID scores closer to 0) and
    higher expression of immunogenicity markers.
    """
    
    return 1