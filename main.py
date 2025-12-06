"""Perform intervention and evaluate with SenCID."""
import matplotlib.pyplot as plt
import model
import env

# Configuration
from configs import SEED, NUM_EPISODES, MAX_STEPS_PER_EPISODE, VERBOSE, HARDOCODED_REWARD, AVAILABLE_GENES_TO_INTERVENE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LR_RATE, EPSILON_DECAY_EPISODES  
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Reward plot
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    line1, = ax1.plot([], [], 'b-', label='Reward', linewidth=2)
    ax1.legend()
    
    # Loss plot
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    line2, = ax2.plot([], [], 'r-', label='Loss', linewidth=2)
    ax2.legend()
    
    # Winning SID plot
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Winning SID')
    ax3.set_title(f'Winning SID per Episode (Target: {TARGET_SID})')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 6.5)
    ax3.set_yticks([1, 2, 3, 4, 5, 6])
    ax3.set_yticklabels(['SID1', 'SID2', 'SID3', 'SID4', 'SID5', 'SID6'])
    line3, = ax3.plot([], [], 'go-', label='Winning SID', linewidth=2, markersize=4)
    # Add target line
    target_idx = int(TARGET_SID.split('SID')[1])
    ax3.axhline(y=target_idx, color='red', linestyle='--', label=f'Target ({TARGET_SID})', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig, ax1, ax2, ax3, line1, line2, line3


def update_plot(fig, ax1, ax2, ax3, line1, line2, line3, episodes, rewards_history, losses_history, sid_history):
    """Update plot with new data."""
    line1.set_data(episodes, rewards_history)
    line2.set_data(episodes, losses_history)
    line3.set_data(episodes, sid_history)
    
    # Adjust axes limits
    ax1.set_xlim(0, max(episodes) + 1)
    ax1.set_ylim(min(rewards_history) - 0.1, max(rewards_history) + 0.1)
    
    ax2.set_xlim(0, max(episodes) + 1)
    ax2.set_ylim(min(losses_history) - 0.1, max(losses_history) + 0.1)
    
    ax3.set_xlim(0, max(episodes) + 1)
    
    # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.1)


def main():
    # Set seed for reproducibility
    
    print(f"Training for {NUM_EPISODES} episodes with seed {SEED}")
    print("AVAILABLE_GENES_TO_INTERVENE", len(AVAILABLE_GENES_TO_INTERVENE))

    # model = REINFORCE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, lr=LR_RATE, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    model = ActorCritic(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, lr=LR_RATE, epsilon_decay_episodes=EPSILON_DECAY_EPISODES)
    # Setup real-time plot
    fig, ax1, ax2, ax3, line1, line2, line3 = setup_plot()
    episodes_list = []
    rewards_history = []
    losses_history = []
    sid_history = []
    
    for episode in range(NUM_EPISODES):
        cell_state = env.reset(cell_index=0)  # Use first cell
        
        log_probs = []
        rewards = []
        values = []  # For ActorCritic
        winning_sid_idx = 1  # default
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Pick action and get log probability (+ value for ActorCritic)
            action_gene, action_type, log_prob, value = model.pick_action(cell_state)
            log_probs.append(log_prob)
            values.append(value)
            
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
        loss = model.update_policy(log_probs, rewards, values)
        model.decay_epsilon()  # Decay epsilon once per episode
        total_reward = sum(rewards)
        print(rewards)
        print(f"Ep {episode+1}: Steps={len(rewards)}, R={total_reward:.3f}, Loss={loss:.4f}, ε={model.epsilon:.3f}, SID={winning_sid_idx}")
        if (winning_sid_idx == 0):
            break
        
        # Update plot
        episodes_list.append(episode + 1)
        rewards_history.append(total_reward)
        losses_history.append(loss)
        sid_history.append(winning_sid_idx)
        update_plot(fig, ax1, ax2, ax3, line1, line2, line3, episodes_list, rewards_history, losses_history, sid_history)
    
    # Save and show plot
    plt.ioff()
    fig.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Plot saved to training_progress.png")
    plt.show()


if __name__ == "__main__":
    main()
