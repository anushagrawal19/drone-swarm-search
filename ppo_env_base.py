from DSSE import DroneSwarmSearch
import numpy as np
import random

class BaseDroneSwarmSearch(DroneSwarmSearch):
    """Base version of DroneSwarmSearch with probability matrix rewards"""

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
        prob_reward_factor=10.0,  # Factor to scale probability rewards
        min_prob_threshold=0.001  # Threshold for considering an area as "high probability"
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

        # Additional parameters for probability rewards
        self.prob_reward_factor = prob_reward_factor
        self.min_prob_threshold = min_prob_threshold

        # Additional metrics tracking
        self.episode_metrics = {
            'successful_searches': 0,
        }

    def get_normalized_observation(self, agent_idx):
        """Convert raw observations to normalized form suitable for PPO"""
        position = self.agents_positions[agent_idx]
        prob_matrix = self.probability_matrix.get_matrix()

        # Flatten and normalize position
        normalized_pos = np.array([
            position[0] / self.grid_size,
            position[1] / self.grid_size
        ])

        return {
            'position': normalized_pos,
            'probability_matrix': prob_matrix,
        }

    def calculate_prob_reward(self, agent_idx, old_pos, new_pos):
        """Calculate reward based on movement towards higher probability areas"""
        prob_matrix = self.probability_matrix.get_matrix()

        # Get the probability values for old and new positions
        old_prob = prob_matrix[old_pos[1], old_pos[0]]
        new_prob = prob_matrix[new_pos[1], new_pos[0]]

        # Reward for moving to a higher probability area
        prob_improvement = new_prob - old_prob
        reward = 0

        if prob_improvement > 0:
            reward += prob_improvement * self.prob_reward_factor
        elif new_prob > self.min_prob_threshold:
            # Small reward for being in a high probability area even if there's no improvement
            reward += 0.1

        return reward

    def step(self, actions):
        """Override step to include probability matrix rewards and metrics tracking"""

        # Store positions before movement
        old_positions = self.agents_positions.copy()

        # Store person positions before movement
        old_person_positions = [(person.x, person.y) for person in self.persons_set]

        # Perform the base step
        observations, rewards, terminations, truncations, infos = super().step(actions)

        # Update metrics for successful searches
        for idx, agent in enumerate(self.agents):
            if agent in rewards:  # Check if agent still active
                if rewards[agent] >= 1:  # If the base reward is greater than 1, the person has been found
                    self.episode_metrics['successful_searches'] += 1

        # Apply probability matrix rewards for each agent
        for idx, agent in enumerate(self.agents):
            if agent in rewards:  # Check if agent still active
                old_pos = old_positions[idx]
                new_pos = self.agents_positions[idx]

                # Calculate the probability matrix-based reward
                prob_reward = self.calculate_prob_reward(idx, old_pos, new_pos)

                # Add the probability reward to the base reward
                rewards[agent] += prob_reward

        # Verify person movement is correct (not towards recharge base)
        for person, old_pos in zip(list(self.persons_set), old_person_positions):
            new_pos = (person.x, person.y)
            if new_pos == old_pos:  # If person hasn't moved, ensure they move according to their vector
                movement_map = self.build_movement_matrix(person)
                person.step(movement_map)

        # Update normalized observations
        normalized_obs = {
            agent: self.get_normalized_observation(idx)
            for idx, agent in enumerate(self.agents)
        }

        return normalized_obs, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """Reset environment and metrics, ensuring drones start at different positions"""
        # Set random but diverse starting positions for drones
        grid_size = self.grid_size
        positions = []

        # Divide grid into regions for each drone
        regions = []
        n_regions = int(np.ceil(np.sqrt(self.drone.amount)))
        region_size = grid_size // n_regions

        for i in range(n_regions):
            for j in range(n_regions):
                if len(regions) < self.drone.amount:
                    regions.append((
                        i * region_size,
                        j * region_size,
                        min((i + 1) * region_size, grid_size),
                        min((j + 1) * region_size, grid_size)
                    ))

        # Place each drone randomly within its region
        for x1, y1, x2, y2 in regions:
            pos = (
                np.random.randint(x1, x2),
                np.random.randint(y1, y2)
            )
            positions.append(pos)

        # Set options for drone positions
        if options is None:
            options = {}
        options['drones_positions'] = positions

        # Set person movement vector
        vector_x = random.uniform(-0.5, 0.5)
        vector_y = random.uniform(-0.5, 0.5)
        options['vector'] = (vector_x, vector_y)

        observations, info = super().reset(seed=seed, options=options)

        # Reset metrics
        self.episode_metrics = {
            'successful_searches': 0,
        }

        # Convert to normalized observations
        normalized_obs = {
            agent: self.get_normalized_observation(idx)
            for idx, agent in enumerate(self.agents)
        }

        return normalized_obs, info
