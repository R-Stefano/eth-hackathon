"""Perform intervention and evaluate with SenCID."""
import sys
import numpy as np
import pandas as pd
import scanpy as sc
sys.path.append('SenCID')
from SenCID.api import SenCID
from intervention import CellState, random_intervention

def main():
    # Load data (genes x cells), then transpose for AnnData (cells x genes)
    counts = pd.read_csv('SenCID/SenCID/demo/demo/origin_matrix_GSE94980.txt', sep='\t', index_col=0).T
    print(f"Loaded: {counts.shape[0]} cells x {counts.shape[1]} genes")
    
    # Sample a cell and create state
    cell_id = np.random.choice(counts.index)
    original_state = CellState(cell_id=cell_id, expression=counts.loc[cell_id])
    print(f"Sampled cell: {cell_id}")
    
    # Apply intervention
    target_genes = ["HLA-B", "TAP1", "CDKN1A", "CDKN2A"]
    intervened_state, intervention = random_intervention(original_state, target_genes)
    print(f"Intervention: {intervention.action} {intervention.gene} by {intervention.magnitude:.2f}x")
    
    # Build AnnData for both states
    orig_adata = sc.AnnData(pd.DataFrame([original_state.expression]))
    orig_adata.obs_names = [original_state.cell_id]
    
    int_adata = sc.AnnData(pd.DataFrame([intervened_state.expression]))
    int_adata.obs_names = [intervened_state.cell_id]
    
    # Run SenCID
    print("\nRunning SenCID...")
    pred_orig, rec_orig, _ = SenCID(orig_adata, denoising=True, binarize=True)
    pred_int, rec_int, _ = SenCID(int_adata, denoising=True, binarize=True)
    
    # Results
    print("\n--- SID Scores ---")
    print(f"{'SID':<6} {'Original':>10} {'Intervened':>12} {'Delta':>10}")
    for sid in pred_orig:
        orig = pred_orig[sid]['SID_Score'].values[0]
        new = pred_int[sid]['SID_Score'].values[0]
        delta = new - orig
        print(f"{sid:<6} {orig:>10.3f} {new:>12.3f} {delta:>+10.3f}")

if __name__ == "__main__":
    main()
