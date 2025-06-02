from DSSE import DroneSwarmSearch
import numpy as np
import random
from gymnasium import spaces

class BaseDroneSwarmSearch2(DroneSwarmSearch):
    """Base version of DroneSwarmSearch with probability matrix rewards - Fixed for PPO convergence"""

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
        prob_reward_factor=100.0,  # Increased for better signal
        min_prob_threshold=0.001,
        exploration_bonus=0.1,  # Small bonus for exploration
        search_bonus=2.0,  # Bonus for searching in high-prob areas
        movement_penalty=0.01,  # Small penalty to encourage efficiency
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

        # Reward parameters
        self.prob_reward_factor = prob_reward_factor
        self.min_prob_threshold = min_prob_threshold
        self.exploration_bonus = exploration_bonus
        self.search_bonus = search_bonus
        self.movement_penalty = movement_penalty

        # Setup observation space for PPO
        self._setup_observation_space()

        # Track visited positions for exploration bonus
        self.visited_positions = set()
        
        # Episode metrics
        self.episode_metrics = {
            'successful_searches': 0,
            'total_reward': 0,
            'steps_taken': 0,
        }

    def _setup_observation_space(self):
        """Setup observation space for PPO training"""
        # Observation components:
        # - Position (2): normalized x, y
        # - Local probability grid (9): 3x3 grid around drone
        # - Global statistics (4): max prob, mean prob, distance to max prob, current prob
        # - Step information (2): timestep ratio, drone index
        
        obs_dim = 2 + 9 + 4 + 2  # Total: 17 dimensions
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_local_probability_grid(self, position, prob_matrix):
        """Get 3x3 probability grid around the drone position"""
        x, y = position
        local_grid = np.zeros(9, dtype=np.float32)
        
        idx = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    local_grid[idx] = prob_matrix[ny, nx]
                idx += 1
                
        # Normalize local grid
        if local_grid.max() > 0:
            local_grid = local_grid / local_grid.max()
            
        return local_grid

    def get_normalized_observation(self, agent_idx):
        """Convert raw observations to normalized form suitable for PPO"""
        if agent_idx >= len(self.agents_positions):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
            
        position = self.agents_positions[agent_idx]
        prob_matrix = self.probability_matrix.get_matrix()
        
        # Normalize position to [0, 1]
        normalized_pos = np.array([
            position[0] / (self.grid_size - 1),
            position[1] / (self.grid_size - 1)
        ], dtype=np.float32)
        
        # Get local probability grid (3x3 around drone)
        local_prob = self._get_local_probability_grid(position, prob_matrix)
        
        # Global probability statistics
        max_prob = np.max(prob_matrix)
        mean_prob = np.mean(prob_matrix)
        current_prob = prob_matrix[position[1], position[0]]
        
        # Find position of maximum probability
        max_pos = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        distance_to_max = np.sqrt((position[0] - max_pos[1])**2 + (position[1] - max_pos[0])**2)
        normalized_distance_to_max = distance_to_max / (self.grid_size * np.sqrt(2))
        
        global_stats = np.array([
            max_prob,
            mean_prob,
            normalized_distance_to_max,
            current_prob
        ], dtype=np.float32)
        
        # Step information
        timestep_ratio = self.timestep / self.timestep_limit if hasattr(self, 'timestep') else 0.0
        drone_index = agent_idx / (self.drone.amount - 1) if self.drone.amount > 1 else 0.0
        
        step_info = np.array([
            timestep_ratio,
            drone_index
        ], dtype=np.float32)
        
        # Combine all observation components
        observation = np.concatenate([
            normalized_pos,      # 2 elements
            local_prob,          # 9 elements
            global_stats,        # 4 elements
            step_info           # 2 elements
        ])
        
        return observation.astype(np.float32)

    def calculate_reward(self, agent_idx, action, old_pos, new_pos):
        """Calculate comprehensive reward based on multiple factors"""
        prob_matrix = self.probability_matrix.get_matrix()
        reward = 0.0
        
        # Get probability values
        old_prob = prob_matrix[old_pos[1], old_pos[0]]
        new_prob = prob_matrix[new_pos[1], new_pos[0]]
        
        # Probability improvement reward
        prob_improvement = new_prob - old_prob
        if prob_improvement > 0:
            reward += prob_improvement * self.prob_reward_factor
        
        # Reward for being in high probability areas
        if new_prob > self.min_prob_threshold:
            reward += new_prob * self.prob_reward_factor * 0.1
            
            # Extra bonus for searching in high probability areas
            if hasattr(self, 'drone') and hasattr(self.drone, 'Actions'):
                if action == self.drone.Actions.SEARCH.value:
                    reward += self.search_bonus * new_prob * 100
        
        # Exploration bonus for visiting new positions
        pos_key = (new_pos[0], new_pos[1])
        if pos_key not in self.visited_positions:
            reward += self.exploration_bonus
            self.visited_positions.add(pos_key)
        
        # Small movement penalty to encourage efficiency
        if old_pos != new_pos:
            reward -= self.movement_penalty
        
        # Penalty for staying in very low probability areas too long
        if new_prob < self.min_prob_threshold * 0.1:
            reward -= 0.05
            
        # Boundary penalty (discourage staying at edges)
        if (new_pos[0] <= 1 or new_pos[0] >= self.grid_size - 2 or 
            new_pos[1] <= 1 or new_pos[1] >= self.grid_size - 2):
            reward -= 0.02
            
        return reward

    def step(self, actions):
        """Override step to include improved rewards and proper observation handling"""
        # Store positions before movement
        old_positions = self.agents_positions.copy()
        
        # Perform the base step
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Modify rewards with our custom reward function
        for idx, agent in enumerate(self.agents):
            if agent in rewards:  # Check if agent still active
                old_pos = old_positions[idx]
                new_pos = self.agents_positions[idx]
                
                # Keep the original reward (for finding person)
                base_reward = rewards[agent]
                
                # Add our custom reward
                custom_reward = self.calculate_reward(idx, actions[agent], old_pos, new_pos)
                
                # Combine rewards
                total_reward = base_reward + custom_reward
                
                # Update metrics
                if base_reward >= 1:  # Person found
                    self.episode_metrics['successful_searches'] += 1
                    total_reward += 100.0  # Large bonus for finding person
                
                rewards[agent] = np.clip(total_reward, -10.0, 110.0)  # Clip rewards
                self.episode_metrics['total_reward'] += rewards[agent]
        
        self.episode_metrics['steps_taken'] += 1
        
        # Convert to normalized observations
        normalized_obs = {}
        for idx, agent in enumerate(self.agents):
            if agent in observations:  # Only create observation if agent is active
                normalized_obs[agent] = self.get_normalized_observation(idx)
        
        return normalized_obs, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """Reset environment with proper initialization"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset visited positions
        self.visited_positions = set()
        
        # Set diverse starting positions for drones
        positions = []
        if self.drone.amount > 1:
            # Divide grid into regions for each drone
            n_regions = int(np.ceil(np.sqrt(self.drone.amount)))
            region_size = max(self.grid_size // n_regions, 3)
            
            regions = []
            for i in range(n_regions):
                for j in range(n_regions):
                    if len(regions) < self.drone.amount:
                        x1 = max(1, i * region_size)
                        y1 = max(1, j * region_size)
                        x2 = min(self.grid_size - 1, (i + 1) * region_size)
                        y2 = min(self.grid_size - 1, (j + 1) * region_size)
                        regions.append((x1, y1, x2, y2))
            
            # Place each drone randomly within its region
            for x1, y1, x2, y2 in regions:
                pos = (
                    np.random.randint(x1, x2),
                    np.random.randint(y1, y2)
                )
                positions.append(pos)
        else:
            # Single drone - random position away from edges
            pos = (
                np.random.randint(2, self.grid_size - 2),
                np.random.randint(2, self.grid_size - 2)
            )
            positions.append(pos)
        
        # Set options
        if options is None:
            options = {}
        options['drones_positions'] = positions
        
        # Random person movement vector
        vector_magnitude = np.random.uniform(0.3, 0.8)
        vector_angle = np.random.uniform(0, 2 * np.pi)
        vector_x = vector_magnitude * np.cos(vector_angle)
        vector_y = vector_magnitude * np.sin(vector_angle)
        options['vector'] = (vector_x, vector_y)
        
        # Call parent reset
        observations, info = super().reset(seed=seed, options=options)
        
        # Reset metrics
        self.episode_metrics = {
            'successful_searches': 0,
            'total_reward': 0,
            'steps_taken': 0,
        }
        
        # Convert to normalized observations
        normalized_obs = {}
        for idx, agent in enumerate(self.agents):
            if agent in observations:
                normalized_obs[agent] = self.get_normalized_observation(idx)
        
        return normalized_obs, info

    def get_episode_metrics(self):
        """Return episode metrics for logging"""
        return self.episode_metrics.copy()