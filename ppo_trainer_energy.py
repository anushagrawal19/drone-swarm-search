import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import argparse

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

class PPOTrainer:
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
        self.run_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.energy_consumed = []
        self.recharge_counts = []
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
            elif key == "energy_consumed":
                self.energy_consumed.append(value)
            elif key == "recharge_count":
                self.recharge_counts.append(value)
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

        # Energy metrics
        axes[1, 0].plot(self.energy_consumed, label='Energy Consumed', alpha=0.7)
        axes[1, 0].plot(self.recharge_counts, label='Recharge Count', alpha=0.7)
        axes[1, 0].set_title('Energy Metrics')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Loss metrics
        if self.policy_losses:
            axes[1, 1].plot(self.policy_losses, label='Policy Loss', alpha=0.7)
            axes[1, 1].plot(self.value_losses, label='Value Loss', alpha=0.7)
            axes[1, 1].set_title('Training Losses')
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
                     len(self.energy_consumed), len(self.recharge_counts),
                     len(self.successful_searches))

        # Pad shorter lists
        def pad_list(lst, target_len):
            return lst + [0] * (target_len - len(lst))

        metrics_array = np.column_stack([
            pad_list(self.episode_rewards, max_len),
            pad_list(self.episode_lengths, max_len),
            pad_list(self.energy_consumed, max_len),
            pad_list(self.recharge_counts, max_len),
            pad_list(self.successful_searches, max_len)
        ])

        np.savetxt(os.path.join(self.run_dir, 'metrics.csv'),
                  metrics_array,
                  delimiter=',',
                  header='rewards,lengths,energy,recharges,targets',
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

    def choose_action_deterministic(self, state):
        """Select action deterministically (for testing)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.network(state_tensor)

            # Take the action with highest probability
            action = torch.argmax(logits, dim=-1)

            return action.item(), 0.0, value.item()

    def save_model(self, filename='model.pt'):
        """Save the current model state"""
        path = os.path.join(self.run_dir, filename)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'energy_consumed': self.energy_consumed,
            'recharge_counts': self.recharge_counts,
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
        self.energy_consumed = checkpoint.get('energy_consumed', [])
        self.recharge_counts = checkpoint.get('recharge_counts', [])
        self.successful_searches = checkpoint.get('successful_searches', [])
        print(f"Model loaded from {path}")

    def test_model(self, n_episodes=100):
        """Test the loaded model and collect metrics"""
        print(f"Testing model for {n_episodes} episodes...")

        # Clear existing metrics to start fresh for testing
        test_rewards = []
        test_lengths = []
        test_energy = []
        test_recharges = []
        test_successes = []

        # Clear for testing
        self.episode_rewards = []
        self.episode_lengths = []
        self.energy_consumed = []
        self.recharge_counts = []
        self.successful_searches = []

        for episode in range(n_episodes):
            episode_reward = 0
            episode_steps = 0

            # Reset environment
            state, _ = self.env.reset()

            # Run episode
            while episode_steps < self.env.timestep_limit:
                # Get actions for all agents
                current_actions = {}

                for agent in self.env.agents:
                    if agent in state:
                        action, _, _ = self.choose_action_deterministic(state[agent])
                        current_actions[agent] = action

                # Take step in environment
                next_state, reward, termination, truncation, _ = self.env.step(current_actions)

                # Accumulate rewards
                for agent in self.env.agents:
                    if agent in reward:
                        episode_reward += reward[agent]

                # Update state
                state = next_state
                episode_steps += 1

                # Check if episode is done
                if any(termination.values()) or any(truncation.values()):
                    break

            # Log episode metrics
            metrics = {
                "episode_reward": episode_reward,
                "episode_length": episode_steps,
                **self.env.episode_metrics
            }
            self.log_metrics(metrics, episode)

            # Store test metrics
            test_rewards.append(episode_reward)
            test_lengths.append(episode_steps)
            test_energy.append(self.env.episode_metrics.get('energy_consumed', 0))
            test_recharges.append(self.env.episode_metrics.get('recharge_count', 0))
            test_successes.append(self.env.episode_metrics.get('successful_searches', 0))

            # Progress update
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(test_rewards[-10:])
                avg_success = np.mean(test_successes[-10:])
                print(f"Test Episode {episode+1:3d}: Reward={episode_reward:6.2f}, "
                      f"Avg10={avg_reward:6.2f}, Success={avg_success:4.2f}, "
                      f"Steps={episode_steps:3d}")

        # Print test statistics
        self.print_test_stats(test_rewards, test_lengths, test_energy,
                             test_recharges, test_successes)

        # Generate test plots
        self.plot_metrics()

        # Save test results
        test_results = {
            "test_episodes": n_episodes,
            "average_reward": float(np.mean(test_rewards)),
            "std_reward": float(np.std(test_rewards)),
            "min_reward": float(np.min(test_rewards)),
            "max_reward": float(np.max(test_rewards)),
            "average_success": float(np.mean(test_successes)),
            "total_success": int(np.sum(test_successes)),
            "average_length": float(np.mean(test_lengths)),
            "average_energy": float(np.mean(test_energy)),
            "average_recharges": float(np.mean(test_recharges)),
            "success_rate": float(np.mean([1 if s > 0 else 0 for s in test_successes]))
        }

        with open(os.path.join(self.run_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)

        return test_results

    def print_test_stats(self, rewards, lengths, energy, recharges, successes):
        """Print test statistics"""
        print(f"\n{'='*50}")
        print(f"TEST RESULTS")
        print(f"{'='*50}")
        print(f"Episodes tested: {len(rewards)}")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Reward range: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
        print(f"Total successful searches: {np.sum(successes)}")
        print(f"Success rate: {np.mean([1 if s > 0 else 0 for s in successes]):.2%}")
        print(f"Average episode length: {np.mean(lengths):.1f} steps")
        print(f"Average energy consumed: {np.mean(energy):.1f}")
        print(f"Average recharges per episode: {np.mean(recharges):.2f}")
        print(f"{'='*50}")

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
            metrics = {
                "episode_reward": episode_reward,
                "episode_length": episode_steps,
                **self.env.episode_metrics
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
        avg_energy = np.mean(self.energy_consumed)
        avg_recharge = np.mean(self.recharge_counts)

        print(f"\n{'='*50}")
        print(f"FINAL TRAINING STATISTICS")
        print(f"{'='*50}")
        print(f"Episodes completed: {len(self.episode_rewards)}")
        print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Reward range: [{min_reward:.2f}, {max_reward:.2f}]")
        print(f"Total successful searches: {total_success}")
        print(f"Average successful searches per episode: {avg_success:.2f}")
        print(f"Average episode length: {avg_length:.1f} steps")
        print(f"Average energy consumed: {avg_energy:.1f}")
        print(f"Average recharges per episode: {avg_recharge:.2f}")

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
            "average_energy_consumed": float(avg_energy),
            "average_recharge_count": float(avg_recharge),
            "run_directory": self.run_dir,
        }

        with open(os.path.join(self.run_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=4)


# Training function
def train_ppo(env, n_episodes=1000, **kwargs):
    """Train PPO on the given environment"""

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
    trainer = PPOTrainer(env, **default_hyperparams)

    # Start training
    trainer.train(n_episodes)

    return trainer


def test_ppo(env, model_path, n_episodes=100, **kwargs):
    """Test a trained PPO model"""

    # Default hyperparameters (should match training)
    default_hyperparams = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "batch_size": 64,
        "n_epochs": 4,
        "log_dir": "test_logs",
        "target_kl": 0.01,
        "normalize_advantages": True,
    }

    # Update with user-provided parameters
    default_hyperparams.update(kwargs)

    # Create trainer
    trainer = PPOTrainer(env, **default_hyperparams)

    # Load the trained model
    trainer.load_model(model_path)

    # Test the model
    test_results = trainer.test_model(
        n_episodes=n_episodes,
    )

    return trainer, test_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test PPO model')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for testing')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes (default: 5000 for train, 100 for test)')

    args = parser.parse_args()

    # Import the fixed environment
    from ppo_env_energy import EnergyAwareDroneSwarmSearch

    if args.mode == 'train':
        print("=== TRAINING MODE ===")

        # Create environment with reasonable parameters
        env = EnergyAwareDroneSwarmSearch(
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
            is_energy=True
        )

        # Set default episodes for training
        n_episodes = args.episodes if args.episodes is not None else 5000

        # Train the model
        trainer = train_ppo(
            env,
            n_episodes=n_episodes,
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=4,
            entropy_coef=0.02,  # Slightly higher for more exploration
        )

        print("Training completed successfully!")
        print(f"Best model saved at: {os.path.join(trainer.run_dir, 'best_model.pt')}")
        print(f"Final model saved at: {os.path.join(trainer.run_dir, 'final_model.pt')}")

    elif args.mode == 'test':
        print("=== TESTING MODE ===")

        if args.model_path is None:
            print("Error: --model_path is required for testing mode")
            print("Example: python ppo_trainer_energy.py --mode test --model_path logs/20241231_120000/best_model.pt")
            exit(1)

        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            exit(1)

        # Create test environment
        env = EnergyAwareDroneSwarmSearch(
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
            is_energy=True
        )

        # Set default episodes for testing
        n_episodes = args.episodes if args.episodes is not None else 100

        print(f"Loading model from: {args.model_path}")
        print(f"Testing for {n_episodes} episodes")

        # Test the model
        trainer, results = test_ppo(
            env,
            args.model_path,
            n_episodes=n_episodes,
        )

        print("Testing completed successfully!")
        print(f"Test results saved at: {os.path.join(trainer.run_dir, 'test_results.json')}")
        print(f"Test plots saved at: {os.path.join(trainer.run_dir, 'training_metrics.png')}")

        # Print summary
        print(f"\n=== TEST SUMMARY ===")
        print(f"Average reward: {results['average_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Success rate: {results['success_rate']:.2%}")

    else:
        print("Invalid mode. Use --mode train or --mode test")
        exit(1)
