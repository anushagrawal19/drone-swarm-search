import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from collections import deque

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
        if n_states == 0:
            return []
        
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

class ImprovedActorCriticNetwork(nn.Module):
    def __init__(self, observation_dim=17, n_actions=5, hidden_dim=128):
        super().__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_actions)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # Ensure state is 2D [batch_size, obs_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        shared_features = self.shared_layers(state)
        
        # Get action logits and state value
        action_logits = self.actor(shared_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic(shared_features)
        
        return action_probs, state_value

class ImprovedPPOTrainer:
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
        target_kl=0.01,  # Early stopping based on KL divergence
        hidden_dim=128,
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

        # Initialize network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get observation dimension from environment
        obs_dim = env.observation_space.shape[0]
        n_actions = len(env.possible_actions)
        print(f"Observation dimension: {obs_dim}, Number of actions: {n_actions}")
        
        self.network = ImprovedActorCriticNetwork(
            observation_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

        # Initialize memory buffer
        self.memory = PPOMemory(batch_size)

        # Setup logging
        self.log_dir = log_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Metrics tracking with moving averages
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_searches = []
        self.value_losses = []
        self.policy_losses = []
        self.kl_divergences = []
        
        # Moving averages for better tracking
        self.reward_ma = deque(maxlen=100)
        self.success_ma = deque(maxlen=100)

        # Save hyperparameters
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon": epsilon,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "grid_size": env.grid_size,
            "drone_amount": env.drone.amount,
            "timestep_limit": env.timestep_limit,
            "hidden_dim": hidden_dim,
            "target_kl": target_kl
        })

    def save_hyperparameters(self, hyperparams):
        """Save hyperparameters to a JSON file"""
        with open(os.path.join(self.run_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hyperparams, f, indent=4)

    def choose_action(self, state):
        """Select action using the current policy"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)
            
            # Ensure probabilities are valid
            action_probs = torch.clamp(action_probs, min=1e-8)
            action_probs = action_probs / action_probs.sum()
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), state_value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value_step = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step * next_non_terminal - values[step]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # lambda = 0.95
            advantages.insert(0, gae)
        
        return advantages

    def update_policy(self):
        """Update policy using PPO with improved stability"""
        if len(self.memory.states) == 0:
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.probs).to(self.device)
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones, dtype=np.float32)
        values = np.array(self.memory.vals)
        
        # Compute GAE
        with torch.no_grad():
            next_value = 0  # Assume episode ends
            advantages = self.compute_gae(rewards, values, dones, next_value)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # Compute returns
            returns = advantages + torch.FloatTensor(values).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Generate batches
        batches = self.memory.generate_batches()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_div = 0
        n_updates = 0
        
        # Update for multiple epochs
        for epoch in range(self.n_epochs):
            for batch_indices in batches:
                if len(batch_indices) == 0:
                    continue
                    
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_probs, state_values = self.network(batch_states)
                
                # Calculate new log probabilities
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio and clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (state_values.squeeze() - batch_returns).pow(2).mean()
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                
                # Calculate KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                    total_kl_div += abs(kl_div)
                
                n_updates += 1
                
                # Early stopping if KL divergence is too high
                if abs(kl_div) > self.target_kl:
                    break
            
            # Early stopping for epoch
            if n_updates > 0 and total_kl_div / n_updates > self.target_kl:
                break
        
        # Store losses for logging
        if n_updates > 0:
            self.policy_losses.append(total_policy_loss / n_updates)
            self.value_losses.append(total_value_loss / n_updates)
            self.kl_divergences.append(total_kl_div / n_updates)

    def train(self, n_episodes):
        """Main training loop with improved convergence"""
        print(f"Starting training... Logs will be saved to {self.run_dir}")
        
        best_reward = float('-inf')
        patience = 200  # Episodes to wait for improvement
        no_improvement_count = 0
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment
            observations, _ = self.env.reset()
            self.memory.clear()
            
            done = False
            
            while not done and episode_steps < self.env.timestep_limit:
                current_actions = {}
                step_rewards = []
                
                # Get actions for each agent
                for agent in self.env.agents:
                    if agent in observations:
                        obs = observations[agent]
                        action, log_prob, value = self.choose_action(obs)
                        current_actions[agent] = action
                        
                        # Store in memory
                        self.memory.store(obs, action, log_prob, value, 0.0, False)
                
                # Step environment
                next_observations, rewards, terminations, truncations, _ = self.env.step(current_actions)
                
                # Update memory with actual rewards and done flags
                agent_idx = 0
                for agent in self.env.agents:
                    if agent in rewards:
                        reward_val = rewards[agent]
                        is_done = terminations.get(agent, False) or truncations.get(agent, False)
                        
                        # Update the stored reward and done flag
                        if len(self.memory.rewards) > agent_idx:
                            self.memory.rewards[-(len(self.env.agents) - agent_idx)] = reward_val
                            self.memory.dones[-(len(self.env.agents) - agent_idx)] = is_done
                        
                        episode_reward += reward_val
                        step_rewards.append(reward_val)
                        agent_idx += 1
                
                observations = next_observations
                done = any(terminations.values()) or any(truncations.values())
                episode_steps += 1
            
            # Update policy after each episode
            if len(self.memory.states) > 0:
                self.update_policy()
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # Get environment metrics
            env_metrics = self.env.get_episode_metrics()
            success_count = env_metrics.get('successful_searches', 0)
            self.successful_searches.append(success_count)
            
            # Update moving averages
            self.reward_ma.append(episode_reward)
            self.success_ma.append(success_count)
            
            # Check for improvement
            current_avg_reward = np.mean(list(self.reward_ma))
            if current_avg_reward > best_reward:
                best_reward = current_avg_reward
                self.save_model('best_model.pt')
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(list(self.reward_ma))
                avg_success = np.mean(list(self.success_ma))
                
                print(f"Episode {episode}: "
                      f"Reward = {episode_reward:.2f}, "
                      f"Avg Reward (100) = {avg_reward:.2f}, "
                      f"Success Rate = {avg_success:.2f}, "
                      f"Steps = {episode_steps}, "
                      f"Best = {best_reward:.2f}")
                
                if len(self.policy_losses) > 0:
                    print(f"  Policy Loss = {self.policy_losses[-1]:.4f}, "
                          f"Value Loss = {self.value_losses[-1]:.4f}, "
                          f"KL Div = {self.kl_divergences[-1]:.4f}")
            
            # Save periodic checkpoints
            if episode % 100 == 0:
                self.save_model(f'checkpoint_episode_{episode}.pt')
                self.plot_metrics()
            
            # Early stopping
            if no_improvement_count >= patience:
                print(f"No improvement for {patience} episodes. Stopping training.")
                break
        
        # Save final model and metrics
        self.save_model('final_model.pt')
        self.plot_metrics()
        self.save_final_metrics()
        
        print(f"\nTraining completed! Results saved to {self.run_dir}")

    def plot_metrics(self):
        """Plot and save training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) > 10:
            ma_rewards = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(self.episode_rewards)), ma_rewards, 
                           color='red', label='Moving Avg (10)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Successful searches
        axes[0, 2].plot(self.successful_searches)
        axes[0, 2].set_title('Successful Searches')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True)
        
        # Policy loss
        if len(self.policy_losses) > 0:
            axes[1, 0].plot(self.policy_losses)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Value loss
        if len(self.value_losses) > 0:
            axes[1, 1].plot(self.value_losses)
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        # KL divergence
        if len(self.kl_divergences) > 0:
            axes[1, 2].plot(self.kl_divergences)
            axes[1, 2].set_title('KL Divergence')
            axes[1, 2].set_xlabel('Update')
            axes[1, 2].set_ylabel('KL Div')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'training_metrics.png'), dpi=150)
        plt.close()

    def save_final_metrics(self):
        """Save final training metrics"""
        if len(self.episode_rewards) == 0:
            return
        
        final_metrics = {
            "training_completed": True,
            "total_episodes": len(self.episode_rewards),
            "average_reward": float(np.mean(self.episode_rewards)),
            "best_reward": float(np.max(self.episode_rewards)),
            "final_100_avg_reward": float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else float(np.mean(self.episode_rewards)),
            "success_rate": float(np.mean(self.successful_searches)) * 100,
            "average_episode_length": float(np.mean(self.episode_lengths)),
            "run_directory": self.run_dir,
        }
        
        with open(os.path.join(self.run_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        print(f"Final metrics: {final_metrics}")

    def save_model(self, filename='model.pt'):
        """Save the current model state"""
        path = os.path.join(self.run_dir, filename)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successful_searches': self.successful_searches,
        }, path)

    def load_model(self, path):
        """Load a saved model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.successful_searches = checkpoint.get('successful_searches', [])
        print(f"Model loaded from {path}")

# Usage example
if __name__ == "__main__":
    # Import your environment
    from ppo_env_base2 import BaseDroneSwarmSearch2
    
    # Create environment with smaller grid for faster training
    env = BaseDroneSwarmSearch2(
        grid_size=20,
        render_mode='ansi',  # Disable rendering for faster training
        render_grid=False,
        render_gradient=False,
        timestep_limit=200,  # Shorter episodes for faster learning
        drone_amount=2,
        drone_speed=5,
        probability_of_detection=0.9,
        prob_reward_factor=50.0,
        exploration_bonus=0.05,
        search_bonus=1.0,
        movement_penalty=0.005,
    )
    
    # Optimized hyperparameters for convergence
    hyperparameters = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "batch_size": 64,
        "n_epochs": 4,
        "target_kl": 0.01,
        "hidden_dim": 128,
        "log_dir": "logs"
    }
    
    # Create trainer and start training
    trainer = ImprovedPPOTrainer(env, **hyperparameters)
    trainer.train(n_episodes=2000)