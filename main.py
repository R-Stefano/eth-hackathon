"""Perform intervention and evaluate with SenCID."""
import matplotlib.pyplot as plt
import model
import env
from collections import Counter

# Configuration
from configs import SEED, NUM_EPISODES, MAX_STEPS_PER_EPISODE, VERBOSE, HARDOCODED_REWARD, AVAILABLE_GENES_TO_INTERVENE, AVAILABLE_ACTIONS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LR_RATE, EPSILON_DECAY_EPISODES, SOURCE_SIDS  
from model import REINFORCE, ActorCritic
import random
import numpy as np
import torch
from configs import TARGET_SID

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def setup_plot():
    """Setup interactive matplotlib plot."""
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Reward plot
    ax2 = fig.add_subplot(gs[0, 1])  # Loss plot
    ax3 = fig.add_subplot(gs[1, 0])  # Final SID plot
    ax4 = fig.add_subplot(gs[1, 1])  # Gradient norm plot
    ax5 = fig.add_subplot(gs[2, :])  # Intervention tracking (full width)
    
    # Reward plot
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    line1, = ax1.plot([], [], 'b-', label='Reward', linewidth=1, alpha=0.4)
    line1_smooth, = ax1.plot([], [], 'b-', label='Reward (MA 50)', linewidth=2)
    ax1.legend()
    
    # Loss plot
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    line2, = ax2.plot([], [], 'r-', label='Loss', linewidth=2)
    ax2.legend()
    
    # Final SID plot
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Final SID')
    ax3.set_title(f'Final SID per Episode (Target: {TARGET_SID})')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 6.5)
    ax3.set_yticks([1, 2, 3, 4, 5, 6])
    ax3.set_yticklabels(['SID1', 'SID2', 'SID3', 'SID4', 'SID5', 'SID6'])
    # Initialize empty scatter plot (will be updated with colors)
    scatter3 = ax3.scatter([], [], s=50, alpha=0.7, label='Final SID')
    # Add target line
    target_idx = int(TARGET_SID.split('SID')[1])
    ax3.axhline(y=target_idx, color='red', linestyle='--', label=f'Target ({TARGET_SID})', alpha=0.7)
    ax3.legend()
    
    # Gradient norm plot (for detecting exploding/vanishing gradients)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Gradient Norm')
    ax4.set_title('Gradient Health Monitor')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale to see both small and large values
    line4, = ax4.plot([], [], 'm-', label='Grad Norm', linewidth=1, alpha=0.4)
    line4_smooth, = ax4.plot([], [], 'm-', label='Grad Norm (MA 50)', linewidth=2)
    # Reference lines for healthy gradient range
    ax4.axhline(y=1e-4, color='orange', linestyle='--', label='Vanishing threshold', alpha=0.7)
    ax4.axhline(y=1e3, color='red', linestyle='--', label='Exploding threshold', alpha=0.7)
    ax4.legend()
    
    # Intervention tracking plot
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Intervention Count')
    ax5.set_title('Intervention Statistics (Top 10 Most Frequent)')
    ax5.grid(True, alpha=0.3)
    # Will be updated with bar chart
    bars5 = None
    
    plt.tight_layout()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig, ax1, ax2, ax3, ax4, ax5, line1, line1_smooth, line2, scatter3, line4, line4_smooth, bars5


def moving_average(data, window=50):
    """Compute moving average with given window size."""
    if len(data) < window:
        # For early episodes, use available data
        cumsum = np.cumsum(data)
        return cumsum / np.arange(1, len(data) + 1)
    else:
        # Full moving average
        cumsum = np.cumsum(data)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        ma = np.zeros(len(data))
        ma[:window] = cumsum[:window] / np.arange(1, window + 1)
        ma[window:] = cumsum[window:] / window
        return ma


def update_plot(fig, ax1, ax2, ax3, ax4, ax5, line1, line1_smooth, line2, scatter3, line4, line4_smooth, bars5,
                episodes, rewards_history, losses_history, sid_history, grad_history, intervention_history):
    """Update plot with new data."""
    line1.set_data(episodes, rewards_history)
    
    # Compute and plot smoothed rewards
    rewards_smooth = moving_average(rewards_history, window=50)
    line1_smooth.set_data(episodes, rewards_smooth)
    
    line2.set_data(episodes, losses_history)
    
    # Update Final SID scatter plot with colors based on target match
    target_idx = int(TARGET_SID.split('SID')[1])
    colors = ['green' if sid == target_idx else 'red' for sid in sid_history]
    # Clear only scatter collections (preserve the target line which is a Line2D)
    for collection in ax3.collections:
        collection.remove()
    scatter3 = ax3.scatter(episodes, sid_history, c=colors, s=50, alpha=0.7, label='Final SID')
    
    # Gradient norm plot
    line4.set_data(episodes, grad_history)
    grad_smooth = moving_average(grad_history, window=50)
    line4_smooth.set_data(episodes, grad_smooth)
    
    # Update intervention statistics
    # Flatten all interventions from all episodes
    all_interventions = []
    for episode_interventions in intervention_history:
        all_interventions.extend(episode_interventions)
    
    if len(all_interventions) > 0:
        # Count interventions
        intervention_counts = Counter(all_interventions)
        # Get top 10 most frequent
        top_interventions = intervention_counts.most_common(10)
        
        # Clear previous bars
        ax5.clear()
        ax5.set_xlabel('Intervention (Gene + Action)')
        ax5.set_ylabel('Count')
        ax5.set_title(f'Intervention Statistics (Top 10 Most Frequent, Total: {len(all_interventions)} interventions)')
        ax5.grid(True, alpha=0.3, axis='y')
        
        if top_interventions:
            labels = [f"{gene}\n{action}" for gene, action in [item[0] for item in top_interventions]]
            counts = [item[1] for item in top_interventions]
            
            bars5 = ax5.barh(range(len(labels)), counts, alpha=0.7, color='steelblue')
            ax5.set_yticks(range(len(labels)))
            ax5.set_yticklabels(labels, fontsize=8)
            ax5.invert_yaxis()  # Top intervention at top
            ax5.set_xlim(0, max(counts) * 1.1)
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars5, counts)):
                ax5.text(count, i, f' {count}', va='center', fontsize=8)
    
    # Adjust axes limits
    ax1.set_xlim(0, max(episodes) + 1 if episodes else 1)
    if rewards_history:
        ax1.set_ylim(min(rewards_history) - 0.1, max(rewards_history) + 0.1)
    
    ax2.set_xlim(0, max(episodes) + 1 if episodes else 1)
    if losses_history:
        ax2.set_ylim(min(losses_history) - 0.1, max(losses_history) + 0.1)
    
    ax3.set_xlim(0, max(episodes) + 1 if episodes else 1)
    
    ax4.set_xlim(0, max(episodes) + 1 if episodes else 1)
    # Auto-adjust y limits for log scale
    if grad_history:
        min_grad = max(min(grad_history), 1e-10)  # Avoid log(0)
        max_grad = max(grad_history)
        ax4.set_ylim(min_grad * 0.1, max_grad * 10)
    
    # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.1)
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    
    return scatter3, bars5  # Return updated scatter and bars for next iteration


def main():
    # Set seed for reproducibility
    
    print(f"Training for {NUM_EPISODES} episodes with seed {SEED}")
    print("AVAILABLE_GENES_TO_INTERVENE", len(AVAILABLE_GENES_TO_INTERVENE))

    # model = REINFORCE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, lr=LR_RATE, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    model = ActorCritic(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, lr=LR_RATE, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    # Setup real-time plot
    fig, ax1, ax2, ax3, ax4, ax5, line1, line1_smooth, line2, scatter3, line4, line4_smooth, bars5 = setup_plot()
    episodes_list = []
    rewards_history = []
    losses_history = []
    sid_history = []
    grad_history = []
    intervention_history = []  # Track interventions per episode
    
    for episode in range(NUM_EPISODES):
        cell_state = env.reset(max_cells_number=40)
        
        log_probs = []
        rewards = []
        values = []  # For ActorCritic
        winning_sid_idx = 1  # default
        episode_interventions = []  # Track interventions for this episode
        
        used_actions = set()  # Track used action indices in this episode
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Pick action and get log probability (+ value for ActorCritic)
            action_gene, action_type, log_prob, value = model.pick_action(cell_state, used_actions)
            log_probs.append(log_prob)
            values.append(value)
            
            # Track intervention
            episode_interventions.append((action_gene, action_type))
            
            # Calculate action index and mark it as used
            action_idx = AVAILABLE_GENES_TO_INTERVENE.index(action_gene) * len(AVAILABLE_ACTIONS) + AVAILABLE_ACTIONS.index(action_type)
            used_actions.add(action_idx)
            
            # Take step in environment
            cell_state_next, sid_scores, terminated = env.perform_step(cell_state, (action_gene, action_type))
            
            # Get winning SID
            winning_sid, winning_sid_idx = env.get_winning_sid(sid_scores)
            
            if VERBOSE:
                print(f"Step {step+1}: {action_gene} {action_type} -> Winning: {winning_sid}")

            # Calculate reward
            reward = env.get_reward(sid_scores)
            if HARDOCODED_REWARD:
                if (action_gene == "HLA-B" and action_type == "ON"):
                    reward = 1.0
                else:
                    reward = -1.0
            rewards.append(reward.item() if hasattr(reward, 'item') else reward)
            
            if terminated:
                print(f"  🎯 SUCCESS! Reached target {TARGET_SID} at step {step+1}")
                break
            cell_state = cell_state_next
        
        # Update policy with collected trajectory
        loss, grad_norm = model.update_policy(log_probs, rewards, values)
        model.decay_epsilon()  # Decay epsilon once per episode
        total_reward = sum(rewards)
        
        # Print intervention summary for this episode
        intervention_summary = Counter(episode_interventions)
        intervention_parts = []
        for (gene, action), count in intervention_summary.items():
            intervention_parts.append(f"{gene} {action}({count})")
        intervention_str = ", ".join(intervention_parts)
        
        print(f"Ep {episode+1}: CellId={cell_state.cell_id}, Interventions={len(intervention_summary)}, R={total_reward:.3f}, Loss={loss:.4f}, GradNorm={grad_norm:.2e}, ε={model.epsilon:.3f}, Output SID={winning_sid_idx}")
        print(f"  Interventions: {intervention_str}")
        
        # Update plot
        episodes_list.append(episode + 1)
        rewards_history.append(total_reward)
        losses_history.append(loss)
        sid_history.append(winning_sid_idx)
        grad_history.append(grad_norm)
        intervention_history.append(episode_interventions)  # Store interventions for this episode
        
        scatter3, bars5 = update_plot(fig, ax1, ax2, ax3, ax4, ax5, line1, line1_smooth, line2, scatter3, line4, line4_smooth, bars5,
                    episodes_list, rewards_history, losses_history, sid_history, grad_history, intervention_history)


if __name__ == "__main__":
    main()
