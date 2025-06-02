from DSSE import DroneSwarmSearch
import numpy as np
from DSSE.environment.constants import Actions
import random

class EnergyAwareDroneSwarmSearch2(DroneSwarmSearch):
    """Energy-aware version of DroneSwarmSearch optimized for PPO convergence"""

    def __init__(
        self,
        grid_size=40,
        render_mode="human",
        render_grid=True,
        render_gradient=True,
        vector=(1, 1),
        timestep_limit=300,
        person_amount=1,
        dispersion_inc=0.05,
        person_initial_position=(15, 15),
        drone_amount=2,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time=0,
        is_energy=True,
        # Normalized reward parameters for better convergence
        energy_penalty_factor=0.1,
        distance_reward_factor=1.0,
        recharge_reward=5.0,  # Reduced from 200
        low_battery_threshold=20,
        base_return_factor=0.5,
        exploration_bonus=0.2,
    ):
        super().__init__(
            grid_size=grid_size,
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            vector=vector,
            timestep_limit=timestep_limit,
            person_amount=person_amount,
            dispersion_inc=dispersion_inc,
            person_initial_position=person_initial_position,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            probability_of_detection=probability_of_detection,
            pre_render_time=pre_render_time,
            is_energy=is_energy,
        )

        # Normalized parameters for better convergence
        self.energy_penalty_factor = energy_penalty_factor
        self.distance_reward_factor = distance_reward_factor
        self.recharge_reward = recharge_reward
        self.low_battery_threshold = low_battery_threshold
        self.base_return_factor = base_return_factor
        self.exploration_bonus = exploration_bonus

        # Tracking for reward normalization
        self.reward_history = []
        self.max_reward_magnitude = 10.0  # Clip rewards to this range

        # Episode metrics
        self.episode_metrics = {
            'energy_consumed': 0,
            'recharge_count': 0,
            'successful_searches': 0,
            'total_reward': 0,
        }

        # Visited cells tracking for exploration bonus
        self.visited_cells = set()

    def get_flattened_observation(self, agent_idx):
        """Return flattened observation vector for PPO compatibility"""
        position = self.agents_positions[agent_idx]
        prob_matrix = self.probability_matrix.get_matrix()
        battery_level = self.drone.get_battery(agent_idx) / 100.0
        
        # Get distance to recharge base (normalized)
        base_pos = self.recharge_base.get_position()
        distance_to_base = np.array([
            (position[0] - base_pos[0]) / self.grid_size,
            (position[1] - base_pos[1]) / self.grid_size
        ])
        
        # Normalized position
        normalized_pos = np.array([
            position[0] / self.grid_size,
            position[1] / self.grid_size
        ])
        
        # Local probability information (5x5 window around drone)
        local_prob = self._get_local_probability(position, prob_matrix)
        
        # Flatten everything into a single vector
        observation = np.concatenate([
            normalized_pos,              # 2 values
            np.array([battery_level]),   # 1 value
            distance_to_base,            # 2 values
            local_prob.flatten(),        # 25 values (5x5 local area)
        ])
        
        return observation.astype(np.float32)

    def _get_local_probability(self, position, prob_matrix, window_size=5):
        """Get local probability matrix around drone position"""
        x, y = position
        half_window = window_size // 2
        
        # Initialize local matrix
        local_matrix = np.zeros((window_size, window_size))
        
        for i in range(window_size):
            for j in range(window_size):
                world_x = x - half_window + i
                world_y = y - half_window + j
                
                # Check bounds
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    local_matrix[j, i] = prob_matrix[world_y, world_x]
        
        return local_matrix

    def calculate_energy_aware_reward(self, agent_idx, base_reward, action, old_pos, new_pos):
        """Calculate normalized energy-aware reward for better PPO convergence"""
        reward = 0.0
        
        # Success reward (target found) - highest priority
        if base_reward >= 1:
            reward += 10.0  # Large positive reward for success
            self.episode_metrics['successful_searches'] += 1
            return np.clip(reward, -self.max_reward_magnitude, self.max_reward_magnitude)
        
        # Get current state information
        battery_level = self.drone.get_battery(agent_idx)
        battery_normalized = battery_level / 100.0
        base_pos = self.recharge_base.get_position()
        prob_matrix = self.probability_matrix.get_matrix()
        
        # Battery management rewards
        if battery_level <= self.low_battery_threshold:
            # Critical battery situation
            if new_pos == base_pos:
                # Successfully recharged
                reward += self.recharge_reward
                self.episode_metrics['recharge_count'] += 1
            else:
                # Encourage moving toward base when battery is low
                old_dist = abs(old_pos[0] - base_pos[0]) + abs(old_pos[1] - base_pos[1])
                new_dist = abs(new_pos[0] - base_pos[0]) + abs(new_pos[1] - base_pos[1])
                
                if new_dist < old_dist:
                    reward += self.base_return_factor  # Moving closer to base
                else:
                    reward -= self.base_return_factor * 2  # Moving away from base (bad!)
        else:
            # Normal operation - encourage exploration and searching
            
            # Reward for searching in high probability areas
            current_prob = prob_matrix[new_pos[1], new_pos[0]]
            if action == Actions.SEARCH.value:
                # Searching reward based on probability
                reward += current_prob * self.distance_reward_factor * 20
            else:
                # Movement rewards
                old_prob = prob_matrix[old_pos[1], old_pos[0]]
                prob_improvement = current_prob - old_prob
                
                # Reward for moving to higher probability areas
                if prob_improvement > 0:
                    reward += prob_improvement * self.distance_reward_factor * 10
                
                # Small exploration bonus for visiting new cells
                cell_key = (new_pos[0], new_pos[1])
                if cell_key not in self.visited_cells:
                    reward += self.exploration_bonus
                    self.visited_cells.add(cell_key)
        
        # Small energy penalty (encourages efficiency)
        energy_penalty = -self.energy_penalty_factor * (1.0 - battery_normalized)
        reward += energy_penalty
        
        # Step penalty to encourage faster completion
        reward -= 0.01
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -self.max_reward_magnitude, self.max_reward_magnitude)
        
        return reward

    def step(self, actions):
        """Override step with improved reward calculation and observation handling"""
        old_positions = self.agents_positions.copy()
        
        # Store person positions before movement
        old_person_positions = [(person.x, person.y) for person in self.persons_set]
        
        # Call parent step
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Ensure person movement is correct
        for person, old_pos in zip(list(self.persons_set), old_person_positions):
            new_pos = (person.x, person.y)
            if new_pos == old_pos:
                movement_map = self.build_movement_matrix(person)
                person.step(movement_map)
        
        # Calculate energy-aware rewards for active agents
        new_rewards = {}
        for idx, agent in enumerate(self.agents):
            if agent in rewards:  # Only process active agents
                old_pos = old_positions[idx]
                new_pos = self.agents_positions[idx]
                
                new_reward = self.calculate_energy_aware_reward(
                    idx, rewards[agent], actions[agent], old_pos, new_pos
                )
                new_rewards[agent] = new_reward
                
                # Update metrics
                self.episode_metrics['total_reward'] += new_reward
                self.episode_metrics['energy_consumed'] += 1
        
        # Get flattened observations for PPO
        flattened_obs = {}
        for idx, agent in enumerate(self.agents):
            if agent in new_rewards:  # Only process active agents
                flattened_obs[agent] = self.get_flattened_observation(idx)
        
        return flattened_obs, new_rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """Reset environment with proper initialization for PPO training"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset visited cells tracking
        self.visited_cells = set()
        
        # Generate diverse starting positions for drones
        positions = self._generate_diverse_positions()
        
        # Set random person movement vector
        vector_x = random.uniform(-1.0, 1.0)
        vector_y = random.uniform(-1.0, 1.0)
        
        if options is None:
            options = {}
        options['drones_positions'] = positions
        options['vector'] = (vector_x, vector_y)
        
        # Call parent reset
        observations, info = super().reset(seed=seed, options=options)
        
        # Reset metrics
        self.episode_metrics = {
            'energy_consumed': 0,
            'recharge_count': 0,
            'successful_searches': 0,
            'total_reward': 0,
        }
        
        # Ensure all drones start with full battery
        for i in range(self.drone.amount):
            self.drone.batteries[i] = self.drone.battery_capacity
        
        # Return flattened observations
        flattened_obs = {}
        for idx, agent in enumerate(self.agents):
            flattened_obs[agent] = self.get_flattened_observation(idx)
        
        return flattened_obs, info

    def _generate_diverse_positions(self):
        """Generate diverse starting positions for drones"""
        positions = []
        grid_size = self.grid_size
        
        # Create a grid of potential starting positions
        n_regions = max(2, int(np.sqrt(self.drone.amount)))
        region_size = grid_size // n_regions
        
        regions = []
        for i in range(n_regions):
            for j in range(n_regions):
                if len(regions) < self.drone.amount:
                    x1 = i * region_size
                    y1 = j * region_size
                    x2 = min((i + 1) * region_size - 1, grid_size - 1)
                    y2 = min((j + 1) * region_size - 1, grid_size - 1)
                    regions.append((x1, y1, x2, y2))
        
        # Place each drone randomly within its region
        for i in range(self.drone.amount):
            if i < len(regions):
                x1, y1, x2, y2 = regions[i]
                pos = (
                    np.random.randint(x1, x2 + 1),
                    np.random.randint(y1, y2 + 1)
                )
            else:
                # Fallback: random position
                pos = (
                    np.random.randint(0, grid_size),
                    np.random.randint(0, grid_size)
                )
            positions.append(pos)
        
        return positions

    def get_observation_space_size(self):
        """Return the size of the flattened observation space"""
        # position(2) + battery(1) + distance_to_base(2) + local_prob(5x5=25)
        return 2 + 1 + 2 + 25