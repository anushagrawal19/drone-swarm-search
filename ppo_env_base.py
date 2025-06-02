from DSSE import DroneSwarmSearch
import numpy as np
import random
from DSSE.environment.constants import Actions

class BaseDroneSwarmSearch2(DroneSwarmSearch):
    """Base version of DroneSwarmSearch with PPO-compatible policy - No energy/battery management"""

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
        is_energy=False,
        # Normalized reward parameters for better convergence
        distance_reward_factor=1.0,
        exploration_bonus=0.2,
        search_bonus_factor=20.0,
        movement_penalty=0.01,
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
            is_energy=is_energy
        )

        # Normalized parameters for better convergence
        self.distance_reward_factor = distance_reward_factor
        self.exploration_bonus = exploration_bonus
        self.search_bonus_factor = search_bonus_factor
        self.movement_penalty = movement_penalty

        # Tracking for reward normalization
        self.max_reward_magnitude = 10.0  # Clip rewards to this range

        # Episode metrics
        self.episode_metrics = {
            'successful_searches': 0,
            'total_reward': 0,
            'steps_taken': 0,
        }

        # Visited cells tracking for exploration bonus
        self.visited_cells = set()

    def get_flattened_observation(self, agent_idx):
        """Return flattened observation vector for PPO compatibility (same structure as EnergyAware but without battery)"""
        position = self.agents_positions[agent_idx]
        prob_matrix = self.probability_matrix.get_matrix()

        # Normalized position
        normalized_pos = np.array([
            position[0] / self.grid_size,
            position[1] / self.grid_size
        ])

        # Local probability information (5x5 window around drone)
        local_prob = self._get_local_probability(position, prob_matrix)

        # Global probability statistics (replacing battery/base distance info)
        max_prob = np.max(prob_matrix)
        mean_prob = np.mean(prob_matrix)
        current_prob = prob_matrix[position[1], position[0]]

        # Flatten everything into a single vector
        # Structure: position(2) + prob_stats(3) + local_prob(25) = 30 total
        observation = np.concatenate([
            normalized_pos,              # 2 values
            np.array([max_prob, mean_prob, current_prob]),  # 3 values (replacing battery info)
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

    def calculate_search_focused_reward(self, agent_idx, base_reward, action, old_pos, new_pos):
        """Calculate normalized search-focused reward for better PPO convergence (no energy considerations)"""
        reward = 0.0

        # Success reward (target found) - highest priority
        if base_reward >= 1:
            reward += 10.0  # Large positive reward for success
            self.episode_metrics['successful_searches'] += 1
            return np.clip(reward, -self.max_reward_magnitude, self.max_reward_magnitude)

        # Get probability information
        prob_matrix = self.probability_matrix.get_matrix()
        old_prob = prob_matrix[old_pos[1], old_pos[0]]
        current_prob = prob_matrix[new_pos[1], new_pos[0]]

        # Reward for searching in high probability areas
        if action == Actions.SEARCH.value:
            # Searching reward based on probability
            reward += current_prob * self.search_bonus_factor
        else:
            # Movement rewards
            prob_improvement = current_prob - old_prob

            # Reward for moving to higher probability areas
            if prob_improvement > 0:
                reward += prob_improvement * self.distance_reward_factor * 10

            # Small exploration bonus for visiting new cells
            cell_key = (new_pos[0], new_pos[1])
            if cell_key not in self.visited_cells:
                reward += self.exploration_bonus
                self.visited_cells.add(cell_key)

        # Small movement penalty to encourage efficiency
        if old_pos != new_pos:
            reward -= self.movement_penalty

        # Step penalty to encourage faster completion
        reward -= 0.01

        # Penalty for staying in very low probability areas
        if current_prob < 0.001:
            reward -= 0.05

        # Boundary penalty (discourage staying at edges)
        if (new_pos[0] <= 1 or new_pos[0] >= self.grid_size - 2 or
            new_pos[1] <= 1 or new_pos[1] >= self.grid_size - 2):
            reward -= 0.02

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

        # Calculate search-focused rewards for active agents
        new_rewards = {}
        for idx, agent in enumerate(self.agents):
            if agent in rewards:  # Only process active agents
                old_pos = old_positions[idx]
                new_pos = self.agents_positions[idx]

                new_reward = self.calculate_search_focused_reward(
                    idx, rewards[agent], actions[agent], old_pos, new_pos
                )
                new_rewards[agent] = new_reward

                # Update metrics
                self.episode_metrics['total_reward'] += new_reward

        self.episode_metrics['steps_taken'] += 1

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
            'successful_searches': 0,
            'total_reward': 0,
            'steps_taken': 0,
        }

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
        # position(2) + prob_stats(3) + local_prob(5x5=25) = 30
        return 2 + 3 + 25

    def get_episode_metrics(self):
        """Return episode metrics for logging"""
        return self.episode_metrics.copy()