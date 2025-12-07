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
        """Decay epsilon linearly to reach epsilon_end at epsilon_decay_episodes."""
        self.episode_count += 1
        progress = min(1.0, self.episode_count / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
        self.episode_count = 0

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

    def pick_action(self, state, used_actions: set = None) -> Tuple[str, str, torch.Tensor]:
        """Pick an action using epsilon-greedy exploration.
        
        Args:
            state: Current cell state
            used_actions: Set of action indices that have already been used in this episode
        """
        if used_actions is None:
            used_actions = set()
        
        state_tensor = torch.tensor(state.expression_norm.values.astype(np.float32).reshape(1, -1))
        probs = self(state_tensor)
        
        # Mask out used actions by setting their probability to 0
        if used_actions:
            mask = torch.ones(self.output_size, dtype=torch.bool)
            for used_idx in used_actions:
                if used_idx < self.output_size:
                    mask[used_idx] = False
            probs = probs * mask.float()
            # Renormalize to ensure valid probability distribution
            probs = probs / (probs.sum() + 1e-9)
        
        dist = torch.distributions.Categorical(probs)
        
        # Epsilon-greedy: random action with probability epsilon
        if random.random() < self.epsilon:
            # EXPLORE: random action from available (unused) actions
            available_actions = [i for i in range(self.output_size) if i not in used_actions]
            if not available_actions:
                # All actions used, allow repeats (shouldn't happen with proper episode length)
                available_actions = list(range(self.output_size))
            action_idx = torch.tensor(random.choice(available_actions))
            if VERBOSE:
                print(f"  [EXPLORE ε={self.epsilon:.3f}] Random action: {action_idx.item()}")
        else:
            # EXPLOIT: sample from policy (already masked)
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
        epsilon_decay_episodes: int = 100,
        dropout: float = 0.1
    ):
        super(ActorCritic, self).__init__()
        
        # 1. Wide Input Projection (Input -> Hidden)
        # Using LayerNorm immediately to handle Z-scored sparse inputs
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.ln_in = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Residual Block 1 (The "Thinking" Block)
        self.res1_fc1 = nn.Linear(hidden_size, hidden_size)
        self.res1_ln1 = nn.LayerNorm(hidden_size)
        self.res1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.res1_ln2 = nn.LayerNorm(hidden_size)
        
        # 3. Residual Block 2 (Deep reasoning)
        self.res2_fc1 = nn.Linear(hidden_size, hidden_size)
        self.res2_ln1 = nn.LayerNorm(hidden_size)
        self.res2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.res2_ln2 = nn.LayerNorm(hidden_size)

        # 4. Residual Block 3 (Deep reasoning)
        self.res3_fc1 = nn.Linear(hidden_size, hidden_size)
        self.res3_ln1 = nn.LayerNorm(hidden_size)
        self.res3_fc3 = nn.Linear(hidden_size, hidden_size)
        self.res3_ln3 = nn.LayerNorm(hidden_size)
        
        # 5. Separate Heads
        self.actor_head = nn.Linear(hidden_size, output_size)
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Activation: GELU is smoother than ReLU and handles negatives better
        self.act = nn.GELU()
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.output_size = output_size
        self.episode_count = 0

    def forward_shared(self, x):
        # Input Projection
        x = self.act(self.ln_in(self.input_proj(x)))
        x = self.dropout(x)
        
        # ResBlock 1
        identity = x
        out = self.act(self.res1_ln1(self.res1_fc1(x)))
        out = self.res1_ln2(self.res1_fc2(out)) # No act on second linear usually in ResNet
        x = self.act(out + identity)  # Add Residual
        
        # ResBlock 2
        identity = x
        out = self.act(self.res2_ln1(self.res2_fc1(x)))
        out = self.res2_ln2(self.res2_fc2(out))
        x = self.act(out + identity)
        
        # ResBlock 3
        identity = x
        out = self.act(self.res3_ln1(self.res3_fc1(x)))
        out = self.res3_ln3(self.res3_fc3(out))
        x = self.act(out + identity)
        
        return x

    def forward(self, x):
        """Returns (action_probs, state_value)."""
        features = self.forward_shared(x)
        
        # Actor
        action_logits = self.actor_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic
        state_value = self.critic_head(features)
        
        return action_probs, state_value
    
    def decay_epsilon(self):
        """Decay epsilon linearly to reach epsilon_end at epsilon_decay_episodes."""
        self.episode_count += 1
        progress = min(1.0, self.episode_count / self.epsilon_decay_episodes)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
    
    def reset_epsilon(self):
        self.epsilon = self.epsilon_start
        self.episode_count = 0
    
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
        
        # Compute gradient norm before optimizer step
        grad_norm = self.get_gradient_norm()
        
        self.optimizer.step()
        
        return loss.item(), grad_norm
    
    def get_gradient_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def pick_action(self, state, used_actions: set = None) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
        """Pick an action using policy sampling with action masking.
        
        For Actor-Critic, we always sample from the policy distribution (not epsilon-greedy).
        This ensures the log_prob matches the action selection method, which is important
        for correct policy gradient updates.
        
        Args:
            state: Current cell state
            used_actions: Set of action indices that have already been used in this episode
        
        Returns: (gene, action_type, log_prob, value)
        """
        if used_actions is None:
            used_actions = set()
        
        state_tensor = torch.tensor(state.expression_norm.values.astype(np.float32).reshape(1, -1))
        probs, value = self(state_tensor)
        
        # Mask out used actions by setting their probability to 0
        if used_actions:
            mask = torch.ones(self.output_size, dtype=torch.bool)
            for used_idx in used_actions:
                if used_idx < self.output_size:
                    mask[used_idx] = False
            probs = probs * mask.float()
            # Renormalize to ensure valid probability distribution
            probs = probs / (probs.sum() + 1e-9)
        
        # Check if all actions are used (shouldn't happen with proper episode length)
        if probs.sum() < 1e-9:
            # Fallback: allow repeats by resetting mask
            probs = self(state_tensor)[0]  # Get original probs
            probs = probs / probs.sum()
        
        dist = torch.distributions.Categorical(probs)
        
        # Always sample from policy distribution (on-policy)
        # The policy distribution already includes exploration through its learned probabilities
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
                
        if VERBOSE:
            entropy = dist.entropy().item()
            print(f"  [Policy Sample] Action: {action_idx.item()} | Entropy: {entropy:.3f} | ε={self.epsilon:.3f}")

        # Decode action
        idx = action_idx.item()
        gene = AVAILABLE_GENES_TO_INTERVENE[idx // len(AVAILABLE_ACTIONS)]
        action_type = AVAILABLE_ACTIONS[idx % len(AVAILABLE_ACTIONS)]
        
        return gene, action_type, log_prob, value.squeeze()





