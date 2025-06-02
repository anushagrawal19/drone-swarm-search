import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super().__init__()

        # Simple but effective architecture for flattened observations
        self.shared_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )

        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        # Process flattened state vector
        shared_features = self.shared_net(state)

        # Get action logits and state value
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)

        return action_logits, state_value.squeeze(-1)

class BasePPOTrainer:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        n_epochs=4,  # Reduced for stability
        log_dir='logs',
        target_kl=0.01,  # Early stopping for KL divergence
        normalize_advantages=True,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        self.normalize_advantages = normalize_advantages

        # Initialize network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Get observation space size from environment
        obs_size = env.get_observation_space_size()
        n_actions = len(env.possible_actions)

        self.network = ActorCriticNetwork(obs_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)

        # Initialize memory buffer
        self.memory = PPOMemory(batch_size)

        # Setup logging
        self.log_dir = log_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"base_env_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_searches = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

        # Save hyperparameters
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon": epsilon,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "obs_size": obs_size,
            "n_actions": n_actions,
            "grid_size": env.grid_size,
            "drone_amount": env.drone.amount,
            "timestep_limit": env.timestep_limit
        })

    def save_hyperparameters(self, hyperparams):
        """Save hyperparameters to a JSON file"""
        with open(os.path.join(self.run_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)

    def log_metrics(self, metrics, episode):
        """Log metrics for the current episode"""
        for key, value in metrics.items():
            if key == "episode_reward":
                self.episode_rewards.append(value)
            elif key == "episode_length":
                self.episode_lengths.append(value)
            elif key == "successful_searches":
                self.successful_searches.append(value)

    def plot_metrics(self):
        """Plot and save training metrics"""
        if len(self.episode_rewards) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.7)
        if len(self.episode_rewards) >= 20:
            ma_rewards = np.convolve(self.episode_rewards, np.ones(20)/20, mode='valid')
            axes[0, 0].plot(range(19, len(self.episode_rewards)), ma_rewards, 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.7)
        if len(self.episode_lengths) >= 20:
            ma_lengths = np.convolve(self.episode_lengths, np.ones(20)/20, mode='valid')
            axes[0, 1].plot(range(19, len(self.episode_lengths)), ma_lengths, 'r-', linewidth=2)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)

        # Successful searches
        axes[0, 2].plot(self.successful_searches, alpha=0.7)
        if len(self.successful_searches) >= 20:
            ma_success = np.convolve(self.successful_searches, np.ones(20)/20, mode='valid')
            axes[0, 2].plot(range(19, len(self.successful_searches)), ma_success, 'r-', linewidth=2)
        axes[0, 2].set_title('Successful Searches per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True)

        # Loss metrics - Policy Loss
        if self.policy_losses:
            axes[1, 0].plot(self.policy_losses, label='Policy Loss', alpha=0.7)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Loss metrics - Value Loss
        if self.value_losses:
            axes[1, 1].plot(self.value_losses, label='Value Loss', alpha=0.7)
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # Success rate over time
        if len(self.successful_searches) >= 50:
            success_rate = np.convolve(self.successful_searches, np.ones(50)/50, mode='valid')
            axes[1, 2].plot(range(49, len(self.successful_searches)), success_rate, 'g-', linewidth=2)
            axes[1, 2].set_title('Success Rate (50-episode moving average)')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Success Rate')
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_metrics.png'), dpi=150)
        plt.close()

        # Save metrics to CSV
        max_len = max(len(self.episode_rewards), len(self.episode_lengths),
                     len(self.successful_searches))

        # Pad shorter lists
        def pad_list(lst, target_len):
            return lst + [0] * (target_len - len(lst))

        metrics_array = np.column_stack([
            pad_list(self.episode_rewards, max_len),
            pad_list(self.episode_lengths, max_len),
            pad_list(self.successful_searches, max_len)
        ])

        np.savetxt(os.path.join(self.run_dir, 'metrics.csv'),
                  metrics_array,
                  delimiter=',',
                  header='rewards,lengths,targets',
                  comments='')

    def choose_action(self, state):
        """Select action using the current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.network(state_tensor)

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample action
            dist = Categorical(probs)
            action = dist.sample()

            return action.item(), dist.log_prob(action).item(), value.item()

    def save_model(self, filename='model.pt'):
        """Save the current model state"""
        path = os.path.join(self.run_dir, filename)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_searches': self.successful_searches,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a saved model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.successful_searches = checkpoint.get('successful_searches', [])
        print(f"Model loaded from {path}")

    def train(self, n_episodes):
        """Main training loop optimized for convergence"""
        print(f"Starting training for {n_episodes} episodes...")
        print(f"Logs will be saved to {self.run_dir}")
        print(f"Observation space size: {self.env.get_observation_space_size()}")
        print(f"Action space size: {len(self.env.possible_actions)}")

        best_reward = float('-inf')
        no_improvement_count = 0
        update_count = 0

        for episode in range(n_episodes):
            episode_reward = 0
            episode_steps = 0

            # Reset environment
            state, _ = self.env.reset()
            self.memory.clear()

            # Collect trajectory
            while episode_steps < self.env.timestep_limit:
                # Get actions for all agents
                current_actions = {}
                agent_states = []
                agent_actions = []
                agent_log_probs = []
                agent_values = []

                for agent in self.env.agents:
                    if agent in state:
                        action, log_prob, value = self.choose_action(state[agent])
                        current_actions[agent] = action

                        # Store for memory
                        agent_states.append(state[agent])
                        agent_actions.append(action)
                        agent_log_probs.append(log_prob)
                        agent_values.append(value)

                # Take step in environment
                next_state, reward, termination, truncation, _ = self.env.step(current_actions)

                # Store experiences
                for i, agent in enumerate(self.env.agents):
                    if agent in reward:
                        self.memory.store(
                            agent_states[i],
                            agent_actions[i],
                            agent_log_probs[i],
                            agent_values[i],
                            reward[agent],
                            termination.get(agent, False) or truncation.get(agent, False)
                        )
                        episode_reward += reward[agent]

                # Update state
                state = next_state
                episode_steps += 1

                # Check if episode is done
                if any(termination.values()) or any(truncation.values()):
                    break

            # Update policy if we have enough experiences
            if len(self.memory.states) > self.batch_size:
                self.update_policy()
                update_count += 1

            # Log episode metrics
            env_metrics = self.env.get_episode_metrics()
            metrics = {
                "episode_reward": episode_reward,
                "episode_length": episode_steps,
                **env_metrics
            }
            self.log_metrics(metrics, episode)

            # Update learning rate
            self.scheduler.step()

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model('best_model.pt')
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Periodic logging and saving
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                avg_success = np.mean(self.successful_searches[-10:]) if len(self.successful_searches) >= 10 else 0
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Episode {episode:4d}: Reward={episode_reward:6.2f}, "
                      f"Avg10={avg_reward:6.2f}, Success={avg_success:4.2f}, "
                      f"Steps={episode_steps:3d}, LR={current_lr:.2e}")

            # Save checkpoints
            if episode % 100 == 0 and episode > 0:
                self.save_model(f'checkpoint_episode_{episode}.pt')
                self.plot_metrics()

        # Save final model and metrics
        self.save_model('final_model.pt')
        self.plot_metrics()

        print(f"\nTraining completed! Results saved to {self.run_dir}")
        self.print_final_stats()

    def update_policy(self):
        """Update policy using PPO algorithm"""
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.probs).to(self.device)
        values = torch.FloatTensor(self.memory.vals).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        dones = torch.BoolTensor(self.memory.dones).to(self.device)

        # Compute returns and advantages
        returns = self.compute_returns(rewards, dones, values)
        advantages = returns - values

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update for multiple epochs
        dataset_size = len(states)

        for epoch in range(self.n_epochs):
            # Generate random indices
            indices = torch.randperm(dataset_size)

            # Process in batches
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]

                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                logits, batch_values = self.network(batch_states)

                # Calculate new log probabilities
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (batch_values - batch_returns).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Store losses for logging
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy_loss.item())

            # Early stopping based on KL divergence
            with torch.no_grad():
                logits, _ = self.network(states)
                new_dist = Categorical(logits=logits)
                new_log_probs = new_dist.log_prob(actions)
                kl_div = (old_log_probs - new_log_probs).mean().item()

                if kl_div > self.target_kl:
                    print(f"Early stopping at epoch {epoch+1} due to KL divergence: {kl_div:.4f}")
                    break

    def compute_returns(self, rewards, dones, values):
        """Compute GAE returns"""
        returns = torch.zeros_like(rewards)
        gae = 0

        # Add final value for incomplete episodes
        next_value = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = values[step + 1]

            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # GAE lambda = 0.95
            returns[step] = gae + values[step]

        return returns

    def print_final_stats(self):
        """Print final training statistics"""
        if len(self.episode_rewards) == 0:
            print("No episodes completed.")
            return

        avg_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        min_reward = np.min(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)

        avg_success = np.mean(self.successful_searches)
        total_success = np.sum(self.successful_searches)

        avg_length = np.mean(self.episode_lengths)

        print(f"\n{'='*50}")
        print(f"FINAL TRAINING STATISTICS")
        print(f"{'='*50}")
        print(f"Episodes completed: {len(self.episode_rewards)}")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Reward range: [{min_reward:.2f}, {max_reward:.2f}]")
        print(f"Total successful searches: {total_success}")
        print(f"Average successful searches per episode: {avg_success:.2f}")
        print(f"Average episode length: {avg_length:.1f} steps")

        # Last 100 episodes performance
        if len(self.episode_rewards) >= 100:
            recent_avg = np.mean(self.episode_rewards[-100:])
            recent_success = np.mean(self.successful_searches[-100:])
            print(f"\nLast 100 episodes:")
            print(f"Average reward: {recent_avg:.2f}")
            print(f"Average success rate: {recent_success:.2f}")

        print(f"{'='*50}")

        # Save final metrics
        final_metrics = {
            "training_completed": True,
            "total_episodes": len(self.episode_rewards),
            "average_reward": float(avg_reward),
            "std_reward": float(std_reward),
            "min_reward": float(min_reward),
            "max_reward": float(max_reward),
            "total_successful_searches": int(total_success),
            "average_successful_searches": float(avg_success),
            "average_episode_length": float(avg_length),
            "run_directory": self.run_dir,
        }

        with open(os.path.join(self.run_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=4)


# Training function
def train_base_ppo(env, n_episodes=1000, **kwargs):
    """Train PPO on the given base environment"""

    # Default hyperparameters optimized for convergence
    default_hyperparams = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "batch_size": 64,
        "n_epochs": 4,
        "log_dir": "logs",
        "target_kl": 0.01,
        "normalize_advantages": True,
    }

    # Update with user-provided parameters
    default_hyperparams.update(kwargs)

    # Create trainer
    trainer = BasePPOTrainer(env, **default_hyperparams)

    # Start training
    trainer.train(n_episodes)

    return trainer


if __name__ == "__main__":
    # Import the base environment
    from ppo_env_base import BaseDroneSwarmSearch

    # Create environment with parameters matching the energy version
    env = BaseDroneSwarmSearch(
        grid_size=20,
        render_mode="ansi",
        render_grid=False,
        render_gradient=False,
        vector=(1, 1),
        timestep_limit=200,
        person_amount=1,
        dispersion_inc=0.05,
        person_initial_position=(10, 10),
        drone_amount=4,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time=0,
        is_energy=False,
    )

    # Train the model with matching parameters
    trainer = train_base_ppo(
        env,
        n_episodes=3000,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=4,
        entropy_coef=0.02,  # Slightly higher for more exploration
    )

    print("Training completed successfully!")