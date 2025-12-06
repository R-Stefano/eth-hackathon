import numpy as np
from typing import Tuple
from intervention import CellState
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)



AVAILABLE_GENES_TO_INTERVENE = ["HLA-B", "TAP1", "CDKN1A", "CDKN2A"]
AVAILABLE_ACTIONS = ["ON", "OFF"]
output_size = len(AVAILABLE_GENES_TO_INTERVENE) * len(AVAILABLE_ACTIONS)
hidden_size = 64
input_size = 1290  # Number of senescence-related genes after ScaleData
model = SimplePolicyNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

def pick_action(state: CellState) -> Tuple[str, float]:
    """Pick a random action for the cell state."""

    probs = model(torch.tensor(state.expression_norm.values.astype(np.float32).reshape(1, -1)))
    # Recover gene and action from model output
    gene_index = torch.multinomial(probs, 1).item()
    gene = AVAILABLE_GENES_TO_INTERVENE[gene_index // len(AVAILABLE_ACTIONS)]
    action = AVAILABLE_ACTIONS[gene_index % len(AVAILABLE_ACTIONS)]
    return gene, action

