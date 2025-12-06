"""Perform intervention and evaluate with SenCID."""
import matplotlib.pyplot as plt
import model
import env

# Configuration
SEED = 42
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 10


def setup_plot():
    """Setup interactive matplotlib plot."""
    plt.ion()  # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
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
    
    plt.tight_layout()
    plt.show(block=False)  # Show window immediately without blocking
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig, ax1, ax2, line1, line2


def update_plot(fig, ax1, ax2, line1, line2, episodes, rewards_history, losses_history):
    """Update plot with new data."""
    line1.set_data(episodes, rewards_history)
    line2.set_data(episodes, losses_history)
    
    # Adjust axes limits
    ax1.set_xlim(0, max(episodes) + 1)
    ax1.set_ylim(min(rewards_history) - 0.1, max(rewards_history) + 0.1)
    
    ax2.set_xlim(0, max(episodes) + 1)
    ax2.set_ylim(min(losses_history) - 0.1, max(losses_history) + 0.1)
    
    # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Longer pause to ensure update is visible


def main():
    # Set seed for reproducibility
    model.set_seed(SEED)
    
    print(f"Training for {NUM_EPISODES} episodes with seed {SEED}")
    
    # Setup real-time plot
    fig, ax1, ax2, line1, line2 = setup_plot()
    episodes_list = []
    rewards_history = []
    losses_history = []
    
    for episode in range(NUM_EPISODES):
        cell_state = env.reset()
        
        log_probs = []
        rewards = []
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Pick action and get log probability
            action_gene, action_type, log_prob = model.pick_action(cell_state)
            log_probs.append(log_prob)
            
            # Take step in environment
            cell_state_next, sid_scores, terminated = env.perform_step(cell_state, (action_gene, action_type))
            
            # Calculate reward
            reward = env.get_reward(sid_scores)
            rewards.append(reward.item() if hasattr(reward, 'item') else reward)
            
            print(f"Episode {episode+1}, Step {step+1}: {action_gene} {action_type}, Reward: {reward:.3f}")
            
            if terminated:
                break
            cell_state = cell_state_next
        
        # Update policy with collected trajectory
        loss = model.update_policy(log_probs, rewards)
        total_reward = sum(rewards)
        print(f"\n=== Episode {episode+1} finished: Total Reward = {total_reward:.3f}, Loss = {loss:.4f} ===\n")
        
        # Update plot
        episodes_list.append(episode + 1)
        rewards_history.append(total_reward)
        losses_history.append(loss)
        update_plot(fig, ax1, ax2, line1, line2, episodes_list, rewards_history, losses_history)
    
    # Save and show plot
    plt.ioff()
    fig.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Plot saved to training_progress.png")
    plt.show()


if __name__ == "__main__":
    main()
