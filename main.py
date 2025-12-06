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
    cell_state = env.reset()
    print(cell_state)

    # Pick and apply intervention
    action_gene, action_type = model.pick_action(cell_state)
    print(f"\nApplying intervention: {action_gene} -> {action_type}")
    
    cell_state_next, sid_scores = env.perform_step(cell_state, (action_gene, action_type))
    
    # Calculate reward
    reward = env.get_reward(sid_scores)
    print(f"\nAction: {action_gene} {action_type}, Reward: {reward:.3f}")
    

if __name__ == "__main__":
    main()
