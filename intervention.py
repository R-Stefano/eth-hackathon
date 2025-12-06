"""
Intervention Pipeline for Senescent Cell State Modulation

Takes single-cell expression data, performs random gene interventions,
and evaluates the resulting cell state using SenCID.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple, Dict
import sys


@dataclass
class CellState:
    """Represents a cell's expression state before/after intervention."""
    cell_id: str
    expression: pd.Series
    intervention: Optional[Dict] = None  # {gene, action, magnitude}


@dataclass 
class Intervention:
    """A single gene intervention."""
    gene: str
    action: Literal["increase", "decrease"]
    magnitude: float  # fold change

def apply_intervention(state: CellState, intervention: Intervention) -> CellState:
    """Apply a gene intervention to a cell state."""
    new_expression = state.expression.copy()
    
    if intervention.gene not in new_expression.index:
        raise ValueError(f"Gene {intervention.gene} not found in expression data")
    
    current_val = new_expression[intervention.gene]
    
    if intervention.action == "increase":
        new_val = current_val * intervention.magnitude
    else:  # decrease
        new_val = current_val / intervention.magnitude
    
    new_expression[intervention.gene] = max(0, new_val)  # expression >= 0
    
    return CellState(
        cell_id=f"{state.cell_id}_intervened",
        expression=new_expression,
        intervention={
            "gene": intervention.gene,
            "action": intervention.action,
            "magnitude": intervention.magnitude,
            "original_value": current_val,
            "new_value": new_expression[intervention.gene]
        }
    )


def random_intervention(state: CellState, gene_list: Optional[List[str]] = None) -> Tuple[CellState, Intervention]:
    """Generate and apply a random intervention based on current expression."""
    # Use provided genes or all genes
    available_genes = gene_list if gene_list else list(state.expression.index)
    
    # Pick random gene
    gene = np.random.choice(available_genes)
    current_expr = state.expression[gene]
    
    # Decide action based on current expression level
    # If expression is high, decrease; if low, increase
    median_expr = state.expression.median()
    action = "decrease" if current_expr > median_expr else "increase"
    
    # Random magnitude between 1.5x and 3x
    magnitude = np.random.uniform(1.5, 3.0)
    
    intervention = Intervention(gene=gene, action=action, magnitude=magnitude)
    new_state = apply_intervention(state, intervention)
    
    return new_state, intervention


