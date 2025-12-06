import numpy as np
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
OUTPUT_SIZE = len(AVAILABLE_GENES_TO_INTERVENE) * len(AVAILABLE_ACTIONS)
HIDDEN_SIZE = 64
INPUT_SIZE = 1290  # Number of senescence-related genes after ScaleData

# Initialize model and optimizer
policy_net = SimplePolicyNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)


def pick_action(state) -> Tuple[str, str, torch.Tensor]:
    """Pick an action and return log probability for training."""
    state_tensor = torch.tensor(state.expression_norm.values.astype(np.float32).reshape(1, -1))
    probs = policy_net(state_tensor)
    
    # Sample action
    dist = torch.distributions.Categorical(probs)
    action_idx = dist.sample()
    log_prob = dist.log_prob(action_idx)
    
    # Decode action
    idx = action_idx.item()
    gene = AVAILABLE_GENES_TO_INTERVENE[idx // len(AVAILABLE_ACTIONS)]
    action_type = AVAILABLE_ACTIONS[idx % len(AVAILABLE_ACTIONS)]
    
    return gene, action_type, log_prob


def update_policy(log_probs: list, rewards: list, gamma: float = 0.99):
    """Update policy using REINFORCE algorithm."""
    # Compute discounted returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    # Normalize returns for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # Compute policy loss
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
    
    return loss.item()

