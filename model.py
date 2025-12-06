import numpy as np
from typing import Tuple
from intervention import CellState

def pick_action(state: CellState) -> Tuple[str, float]:
    """Pick a random action for the cell state."""
    AVAILABLE_GENES_TO_INTERVENE = ["HLA-B", "TAP1", "CDKN1A", "CDKN2A"]
    AVAILABLE_ACTIONS = ["ON", "OFF"]
    gene = np.random.choice(AVAILABLE_GENES_TO_INTERVENE)
    action = np.random.choice(AVAILABLE_ACTIONS)
    return gene, action