import numpy as np
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from configs import VERBOSE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LR_RATE, AVAILABLE_GENES_TO_INTERVENE, AVAILABLE_ACTIONS

class REINFORCE(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        lr: float = LR_RATE,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 100
    ):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Epsilon-greedy exploration
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.output_size = output_size
        self.episode_count = 0
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * (1 - 1 / self.epsilon_decay_episodes))
        self.episode_count += 1
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start

    def update_policy(self, log_probs: list, rewards: list, gamma: float = 0.99):
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
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def pick_action(self, state) -> Tuple[str, str, torch.Tensor]:
        """Pick an action using epsilon-greedy exploration."""
        state_tensor = torch.tensor(state.expression_norm.values.astype(np.float32).reshape(1, -1))
        probs = self(state_tensor)
        dist = torch.distributions.Categorical(probs)
        
        # Epsilon-greedy: random action with probability epsilon
        if random.random() < self.epsilon:
            # EXPLORE: random action
            action_idx = torch.tensor(random.randrange(self.output_size))
            if VERBOSE:
                print(f"  [EXPLORE ε={self.epsilon:.3f}] Random action: {action_idx.item()}")
        else:
            # EXPLOIT: sample from policy
            action_idx = dist.sample()
            if VERBOSE:
                print(f"  [EXPLOIT ε={self.epsilon:.3f}] Policy action: {action_idx.item()}")
        
        log_prob = dist.log_prob(action_idx)
        
        # Decode action
        idx = action_idx.item()
        gene = AVAILABLE_GENES_TO_INTERVENE[idx // len(AVAILABLE_ACTIONS)]
        action_type = AVAILABLE_ACTIONS[idx % len(AVAILABLE_ACTIONS)]

        # Note: epsilon decay moved to per-episode in main.py
        return gene, action_type, log_prob


class ActorCritic(nn.Module):
    """Actor-Critic with separate policy (actor) and value (critic) heads."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float = LR_RATE,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 100
    ):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Epsilon-greedy exploration
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.output_size = output_size
        self.episode_count = 0
    
    def forward(self, x):
        """Returns (action_probs, state_value)."""
        features = self.shared(x)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * (1 - 1 / self.epsilon_decay_episodes))
        self.episode_count += 1
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
    
    def update_policy(self, log_probs: list, rewards: list, values: list, gamma: float = 0.99, 
                      value_coef: float = 0.5, entropy_coef: float = 0.01):
        """Update policy using Advantage Actor-Critic."""
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        # Advantage = Returns - Value estimates (baseline subtraction)
        advantages = returns - values.detach()
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
        
        # Policy loss (actor): -log_prob * advantage
        policy_loss = -(log_probs * advantages).sum()
        
        # Value loss (critic): MSE between predicted values and returns
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus for exploration (optional)
        # entropy = -(probs * log_probs).sum()  # Would need probs stored
        
        # Combined loss
        loss = policy_loss + value_coef * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def pick_action(self, state) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
        """Pick an action using epsilon-greedy exploration.
        
        Returns: (gene, action_type, log_prob, value)
        """
        state_tensor = torch.tensor(state.expression_norm.values.astype(np.float32).reshape(1, -1))
        probs, value = self(state_tensor)
        dist = torch.distributions.Categorical(probs)
        
        # Epsilon-greedy: random action with probability epsilon
        if random.random() < self.epsilon:
            action_idx = torch.tensor(random.randrange(self.output_size))
            if VERBOSE:
                print(f"  [EXPLORE ε={self.epsilon:.3f}] Random action: {action_idx.item()}")
        else:
            action_idx = dist.sample()
            if VERBOSE:
                print(f"  [EXPLOIT ε={self.epsilon:.3f}] Policy action: {action_idx.item()}")
        
        log_prob = dist.log_prob(action_idx)
        
        # Decode action
        idx = action_idx.item()
        gene = AVAILABLE_GENES_TO_INTERVENE[idx // len(AVAILABLE_ACTIONS)]
        action_type = AVAILABLE_ACTIONS[idx % len(AVAILABLE_ACTIONS)]
        
        return gene, action_type, log_prob, value.squeeze()





