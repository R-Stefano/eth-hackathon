"""Perform intervention and evaluate with SenCID."""
import sys
import numpy as np
import pandas as pd
import scanpy as sc
sys.path.append('SenCID')
from SenCID.api import SenCID
from intervention import CellState
import model
import env

def main():
    # Load data (genes x cells), then transpose for AnnData (cells x genes)
    counts = pd.read_csv('SenCID/SenCID/demo/demo/origin_matrix_GSE94980.txt', sep='\t', index_col=0).T
    print(f"Loaded: {counts.shape[0]} cells x {counts.shape[1]} genes")
    
    # Sample a cell and create state
    cell_id = np.random.choice(counts.index)
    cell_state = CellState(cell_id=cell_id, expression=counts.loc[cell_id])
    print(f"Sampled cell: {cell_id}")

    # # Get original SID scores
    # print("\nRunning SenCID on original cell...")
    # orig_adata = sc.AnnData(pd.DataFrame([cell_state.expression]))
    # orig_adata.obs_names = [cell_state.cell_id]
    # pred_orig, rec_orig, _ = SenCID(orig_adata, denoising=False, binarize=True)

    # Pick and apply intervention
    action_gene, action_type = model.pick_action(cell_state)
    print(f"\nApplying intervention: {action_gene} -> {action_type}")
    
    cell_state_next, sid_scores = env.perform_step(cell_state, (action_gene, action_type))
    
    # Calculate reward
    reward = env.get_reward(sid_scores)
    print(f"\nAction: {action_gene} {action_type}, Reward: {reward:.3f}")
    

if __name__ == "__main__":
    main()
