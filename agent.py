"""Simple PPO agent to modify a single-cell expression vector to maximize a target SID class.

This file provides:
- A PyTorch policy+value network that maps a single-cell gene-expression vector
  to a categorical distribution over (num_genes * 3) actions: for each gene
  {do-nothing, upregulate, downregulate}.
- A small PPO trainer that runs episodes where after each action we modify the
  input cell expression (in z-scored space), re-run the SenCID SID predictors
  (via the repository's `SenCID` package functions) and compute a reward defined
  as the negative cross-entropy between the SID softmax distribution and the
  desired one-hot target class. The episode stops when the desired-class
  probability exceeds a threshold.

Notes / assumptions:
- This implementation expects the `SenCID` package to be importable from the
  current Python path (the repository's `SenCID` directory). It calls
  `SenCID.DataPro.ScaleData` and `SenCID.Pred.Pred` to obtain SID scores.
- The change applied when up/down-regulating a gene is additive in the
  z-scored space; default change magnitude is 1.0 (one standard deviation).
- The code is intentionally minimal and focused on clarity rather than
  performance or distributed training. It is suitable for single-cell /
  single-episode debugging and local training.

Example (high-level):
	from agent import PPOAgent, PPOTrainer
	# Prepare a single cell (pandas Series with gene-index) using SenCID.DataPro.ScaleData
	trainer = PPOTrainer(agent, ...)
	result = trainer.run_episode(cell_series, desired_sid=2)

"""

from typing import Tuple
import os
import sys
import math
import copy
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Try to import the SenCID functions used to compute SID scores
try:
	from SenCID.DataPro import ScaleData
	from SenCID.Pred import Pred
except Exception:
	# If running from repository root, add the local SenCID folder to path
	repo_root = os.path.dirname(__file__)
	candidate = os.path.join(repo_root, 'SenCID')
	if os.path.isdir(candidate):
		sys.path.append(repo_root)
	try:
		from SenCID.DataPro import ScaleData
		from SenCID.Pred import Pred
	except Exception as e:
		raise ImportError("Could not import SenCID.DataPro or SenCID.Pred. "
						  "Make sure the SenCID package is available in PYTHONPATH.") from e


class PolicyValueNet(nn.Module):
	"""Shared backbone with separate policy and value heads."""
	def __init__(self, input_dim: int, num_genes: int, hidden_sizes=(256, 128)):
		super().__init__()
		self.input_dim = input_dim
		self.num_genes = num_genes
		# simple MLP backbone
		layers = []
		last = input_dim
		for h in hidden_sizes:
			layers.append(nn.Linear(last, h))
			layers.append(nn.ReLU())
			last = h
		self.backbone = nn.Sequential(*layers)
		# policy head: outputs logits over num_genes * 3 actions
		self.policy_head = nn.Linear(last, num_genes * 3)
		# value head: scalar
		self.value_head = nn.Linear(last, 1)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return (logits, value) for input x shaped (batch, input_dim)."""
		h = self.backbone(x)
		logits = self.policy_head(h)
		value = self.value_head(h).squeeze(-1)
		return logits, value


class PPOAgent:
	def __init__(self, gene_names, lr=3e-4, device=None, hidden_sizes=(256,128)):
		"""Create agent.

		gene_names: list of genes (order must match the preprocessed input vector)
		"""
		self.gene_names = list(gene_names)
		self.num_genes = len(self.gene_names)
		self.input_dim = self.num_genes
		self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
		self.net = PolicyValueNet(self.input_dim, self.num_genes, hidden_sizes).to(self.device)
		self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

	def act(self, x: np.ndarray) -> Tuple[int, float, float]:
		"""Sample an action given a single observation x (numpy array shape (input_dim,)).

		Returns (action_index, log_prob, value)
		"""
		self.net.eval()
		with torch.no_grad():
			tx = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
			logits, value = self.net(tx)
			probs = torch.softmax(logits, dim=-1)
			dist = Categorical(probs)
			action = dist.sample()
			logp = dist.log_prob(action)
		return int(action.item()), float(logp.item()), float(value.item())

	def get_logits_and_value(self, x_batch: torch.Tensor):
		return self.net(x_batch)


class PPOTrainer:
	def __init__(self, agent: PPOAgent,
				 clip_epsilon=0.2, vf_coef=0.5, ent_coef=0.01,
				 gamma=0.99, lam=0.95, epochs=10, batch_size=64):
		self.agent = agent
		self.clip_epsilon = clip_epsilon
		self.vf_coef = vf_coef
		self.ent_coef = ent_coef
		self.gamma = gamma
		self.lam = lam
		self.epochs = epochs
		self.batch_size = batch_size

	@staticmethod
	def _action_to_gene_and_mode(action_idx: int, num_genes: int) -> Tuple[int,int]:
		gene = action_idx // 3
		mode = action_idx % 3  # 0 noop, 1 up, 2 down
		return gene, mode

	def _apply_action(self, cell_series: pd.Series, action_idx: int, change_amount: float = 1.0) -> pd.Series:
		"""Return a new cell_series with the selected gene modified.

		We operate in z-scored space (the preprocessed space returned by ScaleData).
		change_amount is added/subtracted to the selected gene's value.
		"""
		gene_idx, mode = self._action_to_gene_and_mode(action_idx, self.agent.num_genes)
		gene = self.agent.gene_names[gene_idx]
		new_cell = cell_series.copy()
		if mode == 1:  # upregulate
			new_cell.loc[gene] = new_cell.loc.get(gene, 0.0) + change_amount
		elif mode == 2:  # downregulate
			new_cell.loc[gene] = new_cell.loc.get(gene, 0.0) - change_amount
		# mode == 0 means no change
		return new_cell

	def _compute_sid_probs(self, cell_series: pd.Series, sidnums=(1,2,3,4,5,6)) -> np.ndarray:
		"""Given a single-cell series indexed by gene names (z-scored space),
		compute SID scores (one per SID) and return a softmax-normalized numpy array.
		"""
		# Build cpm_zcol DataFrame expected by Pred: rows are genes, columns are cells
		cpm_zcol = pd.DataFrame(cell_series.values.reshape(-1,1), index=cell_series.index, columns=['cell_0'])
		sid_scores = []
		for sid in sidnums:
			labels = Pred(cpm_zcol, sid, binarize=False)
			# labels is a DataFrame indexed by cell with column 'SID_Score'
			score = float(labels.loc['cell_0', 'SID_Score'])
			sid_scores.append(score)
		sid_scores = np.array(sid_scores, dtype=np.float32)
		# convert to probabilities via softmax
		exp = np.exp(sid_scores - sid_scores.max())
		probs = exp / exp.sum()
		return probs

	def _reward_from_probs(self, probs: np.ndarray, desired_idx: int) -> float:
		"""Reward is negative cross-entropy between probs and one-hot(desired_idx).
		Higher reward when more mass on desired class.
		"""
		eps = 1e-8
		ce = -math.log(max(probs[desired_idx], eps))  # cross-entropy with one-hot is -log p_target
		return -ce

	def run_episode(self, initial_cell: pd.Series, desired_sid: int, threshold: float = 0.9,
					max_steps: int = 50, change_amount: float = 1.0, render: bool=False):
		"""Run a single episode starting from initial_cell (pandas Series indexed by genes).

		Returns a dict with trajectory and final info.
		"""
		# sanity: ensure gene order matches agent
		cell = initial_cell.reindex(self.agent.gene_names).fillna(0.0)
		traj = {'obs': [], 'actions': [], 'logps': [], 'values': [], 'rewards': []}

		for step in range(max_steps):
			x = cell.values.astype(np.float32)
			action_idx, logp, value = self.agent.act(x)
			new_cell = self._apply_action(cell, action_idx, change_amount=change_amount)
			# compute SID probs
			probs = self._compute_sid_probs(new_cell)
			reward = self._reward_from_probs(probs, desired_idx=desired_sid-1)

			traj['obs'].append(x)
			traj['actions'].append(action_idx)
			traj['logps'].append(logp)
			traj['values'].append(value)
			traj['rewards'].append(reward)

			cell = new_cell

			p_desired = float(probs[desired_sid-1])
			if render:
				print(f"step={step} action={action_idx} p_desired={p_desired:.4f} reward={reward:.4f}")
			if p_desired >= threshold:
				break

		result = {
			'traj': traj,
			'final_cell': cell,
			'final_probs': probs,
			'steps': len(traj['rewards']),
			'success': float(probs[desired_sid-1]) >= threshold,
		}
		return result

	def update(self, trajectories: list):
		"""Update policy using a list of trajectory dicts produced by run_episode.

		Each trajectory dict must contain 'obs', 'actions', 'logps', 'values', 'rewards'.
		"""
		# flatten
		obs = np.concatenate([np.array(t['obs'], dtype=np.float32) for t in trajectories], axis=0)
		actions = np.concatenate([np.array(t['actions'], dtype=np.int64) for t in trajectories], axis=0)
		old_logps = np.concatenate([np.array(t['logps'], dtype=np.float32) for t in trajectories], axis=0)
		values = np.concatenate([np.array(t['values'], dtype=np.float32) for t in trajectories], axis=0)
		rewards = [t['rewards'] for t in trajectories]

		# compute discounted returns and advantages per-trajectory
		returns = []
		advantages = []
		ptr = 0
		for r_seq, v_seq in zip(rewards, [t['values'] for t in trajectories]):
			# compute returns
			R = 0.0
			ret_seq = []
			for r in reversed(r_seq):
				R = r + self.gamma * R
				ret_seq.insert(0, R)
			ret_seq = np.array(ret_seq, dtype=np.float32)
			# advantage = returns - values
			v_seq = np.array(v_seq, dtype=np.float32)
			adv = ret_seq - v_seq
			returns.append(ret_seq)
			advantages.append(adv)
			ptr += len(r_seq)

		returns = np.concatenate(returns, axis=0)
		advantages = np.concatenate(advantages, axis=0)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		# convert to tensors
		obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.agent.device)
		actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.agent.device)
		old_logps_t = torch.as_tensor(old_logps, dtype=torch.float32, device=self.agent.device)
		returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.agent.device)
		advs_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.agent.device)

		n = obs_t.shape[0]
		inds = np.arange(n)
		for _ in range(self.epochs):
			np.random.shuffle(inds)
			for start in range(0, n, self.batch_size):
				end = start + self.batch_size
				batch_idx = inds[start:end]
				b_obs = obs_t[batch_idx]
				b_actions = actions_t[batch_idx]
				b_oldlogp = old_logps_t[batch_idx]
				b_returns = returns_t[batch_idx]
				b_advs = advs_t[batch_idx]

				logits, values = self.agent.get_logits_and_value(b_obs)
				probs = torch.softmax(logits, dim=-1)
				dist = Categorical(probs)
				new_logp = dist.log_prob(b_actions)
				entropy = dist.entropy().mean()

				ratio = torch.exp(new_logp - b_oldlogp)
				surr1 = ratio * b_advs
				surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advs
				policy_loss = -torch.min(surr1, surr2).mean()

				value_loss = (b_returns - values).pow(2).mean()

				loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

				self.agent.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.agent.net.parameters(), max_norm=0.5)
				self.agent.optimizer.step()

		return {
			'policy_loss': float(policy_loss.item()),
			'value_loss': float(value_loss.item()),
			'entropy': float(entropy.item())
		}


if __name__ == '__main__':
	# simple smoke test / usage example (requires SenCID models and data available)
	# This block will not be executed if the module is imported.
	# The user must prepare a pandas Series `cell_series` containing the z-scored
	# values for the SenCID seneset genes (index order will be reindexed to agent.gene_names).
	try:
		# determine gene list from the DataPro resource by running ScaleData on a toy AnnData
		# Here we attempt to create a dummy AnnData from the resource genes to instantiate the agent.
		sene = pd.read_csv(os.path.join(os.path.dirname(__file__), 'SenCID', 'resource', 'seneset.txt'), sep='\t', header=None)
		gene_list = list(sene[0].astype(str).unique())
		agent = PPOAgent(gene_list)
		trainer = PPOTrainer(agent)
		print('Agent created with', len(gene_list), 'genes. Define a real cell Series and call trainer.run_episode(...)')
	except Exception:
		print('Could not auto-create agent; ensure SenCID/resource/seneset.txt is available and the SenCID package imports correctly.')

