#!/usr/bin/env python3

from enum import Enum
import math
from MDP.mdp import MDP
import numpy as np
import matplotlib.pyplot as plt

class Actions(Enum):
    UP=1
    DOWN=2
    LEFT=3
    RIGHT=4

_VEC = {
    Actions.UP: np.array([0, 1]),
    Actions.DOWN: np.array([0, -1]),
    Actions.LEFT: np.array([-1, 0]),
    Actions.RIGHT: np.array([1, 0])
}

_ACTION_VECS = {
    Actions.UP: np.array([1, 0, 0, 0]),
    Actions.DOWN: np.array([0, 1, 0, 0]),
    Actions.LEFT: np.array([0, 0, 1, 0]),
    Actions.RIGHT: np.array([0, 0, 0, 1])
}

def _clip(p, max_x, max_y):
    p = np.array([max(min(p[0], max_x), 0),
                  max(min(p[1], max_y), 0)])
    return p


class Obstacle:
    def __init__(self, x, y, radius, reward):
        self.pos = np.array([x, y])
        self.rad = radius
        self.reward = reward

    def contains(self, pos2):
        return np.linalg.norm(self.pos-pos2) <= self.rad


class ContinuousGridWorldMDP(MDP):
    def __init__(self, w, h, move_cost=-0.1, default_start=None):
        self.width = w
        self.height = h
        self.move_cost = move_cost
        self.goal_x = None
        self.goal_y = None
        self.walls = []
        self._obs = []
        self.start_pos = default_start if default_start is not None else [0.0, 0.0]

    def add_pit(self, x, y, radius, cost=10.0):
        self._obs += [Obstacle(x, y, radius, -cost)]

    def add_goal(self, x, y, radius, reward=10.0):
        self.goal_x = x
        self.goal_y = y
        self._obs += [Obstacle(x, y, radius, reward)]

    def add_wall(self, x, y, radius):
        self.walls += [Obstacle(x, y, radius, 0)]

    @property
    def actions(self):
        """Return iterable of all actions."""
        return list(Actions)

    def actions_at(self, state):
        """Return iterable of all actions at given state."""
        return list(Actions)
    
    def discretize(self, state, grid_size=50):
        """
        Discretize the state into a tuple of integers representing the grid position.
        The `grid_size` parameter controls the precision of the discretization.
        """
        # Normalize the state (scaling it between 0 and 1, assuming a 2D grid world)
        x, y = state
        normalized_x = min(max(x, 0), self.width) / self.width
        normalized_y = min(max(y, 0), self.height) / self.height
        
        # Scale by the grid size and clip to valid grid values
        grid_x = int(normalized_x * grid_size)
        grid_y = int(normalized_y * grid_size)
        
        # Ensure we stay within the grid bounds
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)
        
        return (grid_x, grid_y)

    @property
    def initial_state(self):
        """Returns initial state (assumed determinstic)."""
        return np.array(self.start_pos)

    def _in_obs(self, state):
        for obs in self._obs:
            if obs.contains(state):
                return obs
        return None
    
    def r(self, s1, s2):
        """Returns the reward for transitioning from s1 to s2. For now, assume it is deterministic."""
        assert self.goal_x is not None and self.goal_y is not None, "Goal position must be set before calculating rewards."
        goal_pos = np.array([self.goal_x, self.goal_y])  # Set your goal coordinates here
        dist_s2_goal = np.linalg.norm(s2 - goal_pos)
        dist_s1_goal = np.linalg.norm(s1 - goal_pos)

        # Terminal goal condition
        if dist_s2_goal < 1.0:
            return 20.0  # Big reward for reaching the goal

        # Reward for being closer to the goal
        reward = -0.1  # Base cost for moving
        if dist_s2_goal < dist_s1_goal:
            reward += 1  # Reward for getting closer
        elif dist_s2_goal > dist_s1_goal:
            reward -= 1  # Penalty for moving away

        # Add any existing penalties (like hitting pits or walls)
        obs = self._in_obs(s2)
        if obs:
            reward += obs.reward  # Apply obstacle reward (if negative, it's a penalty)

        return reward

    def is_terminal(self, state):
        """Returns true if state s is terminal."""
        if self._in_obs(state):
            return True

        return False ## not great coding, but idc

    def in_walls(self, state):
        """Returns true if state s is in a wall."""
        for wall in self.walls:
            if wall.contains(state):
                return True
        return False

    def act(self, state, action):
        """Observe a single MDP transition."""
        mean = _VEC[action]
        cov = np.eye(2) / 4 + np.abs(np.diag(mean) / 2)
        next_state = state + np.random.multivariate_normal(mean, cov)
        next_state = _clip(next_state, self.width-1, self.height-1)
        if self.in_walls(next_state): # Bounce back
            return state, self.r(state, state)
        return next_state, self.r(state, next_state)
    
    def compute_distance(self, agent_pos):
        x_agent, y_agent = agent_pos
        return math.sqrt((x_agent - self.goal_x) ** 2 + (y_agent - self.goal_y) ** 2)

    def normalize_distance(self, agent_pos):
        max_distance = math.sqrt((self.width - 1) ** 2 + (self.height - 1) ** 2)
        distance = self.compute_distance(agent_pos)
        return distance / max_distance  # Normalized distance in the range [0, 1]

    def normalize_distance_to_nearest_pit(self, agent_pos):
        """
        Computes the normalized distance to the nearest pit.
        The distance is normalized to be in the range [0, 1].
        """
        if not self._obs:
            return 0.0
        # Find the nearest pit
        distances = [np.linalg.norm(agent_pos - obs.pos) for obs in self._obs if obs.reward < 0]
        if len(distances) == 0:
            return 0.0
        min_distance = min(distances)
        return min_distance / (math.sqrt((self.width - 1) ** 2 + (self.height - 1) ** 2))  # Normalized distance to the nearest pit

    def is_action_toward_goal(self, state, action):
        """
        Check if the action is toward the goal.
        Utilizing the dot product of the action vector and the difference between the goal and the current state (as a vector).
        """
        x, y = state
        dx, dy = _ACTION_VECS[action][0], _ACTION_VECS[action][1]
        gx, gy = self.goal_x, self.goal_y

        to_goal_vec = np.array([gx - x, gy - y])
        action_vec = np.array([dx, dy])

        if np.linalg.norm(to_goal_vec) == 0:
            return 1  # We're already at the goal

        # Note that a*b = |a||b|cos(theta) - and we want theta to be small (between 0 and 45 degrees)
        # Compute the dot product
        dot_prod = np.dot(to_goal_vec, action_vec)
        # Compute the magnitudes
        mag_prod = np.linalg.norm(to_goal_vec) * np.linalg.norm(action_vec) + 1e-8 # avoid division by zero
        # Compute the cosine similarity
        cos_theta = dot_prod / mag_prod
        # Check if the action is toward the goal
        if cos_theta > 0.7:  # Adjust threshold as needed - if cos(theta) > 0.7, then theta is between 0 and 45 degrees
            return 1
        else:
            return 0

    def feature_vector(self, state, action):
        """
        Returns the feature vector for a given state and action.
        The feature vector is a list of features that represent the state-action pair.
        """
        # Example feature vector: [x, y, action]
        assert self.goal_x is not None and self.goal_y is not None, "Goal position must be set before calculating feature vector."
        x, y = state
        distance_to_goal = self.normalize_distance(state)
        inv_goal_dist = 1 / (distance_to_goal + 1e-5)  # Big value near goal
        distance_to_pit = self.normalize_distance_to_nearest_pit(state)
        toward_goal = int(self.is_action_toward_goal(state, action))
        action_vec = _ACTION_VECS[action]
        return np.array([
            distance_to_goal,
            inv_goal_dist,
            distance_to_pit,
            toward_goal,
            x / self.width,
            y / self.height,
            action_vec[0],
            action_vec[1]
        ])

    def display(self, states, policy_algorithm=None):
        """
        Helper method to display the gridworld after an agent has traversed a certain path through it
        """
        _, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        # Create the grid and set labels
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xticklabels(range(self.width))
        ax.set_yticklabels(range(self.height))
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

        # Plot agent's path with a thicker line and dashed segments for sharper turns
        x = [state[0] for state in states]
        y = [state[1] for state in states]
        ax.plot(x, y, 'bo-', markersize=5, linewidth=2)

        # Plot start and end positions
        start_state = states[0] if states else None
        end_state = states[-1] if states else None
        if start_state is not None:
            ax.scatter(start_state[0], start_state[1], color='blue', s=100, label="Start", marker='*')
        if end_state is not None:
            ax.scatter(end_state[0], end_state[1], color='red', s=100, label="End", marker='*')

        # Now show all of the obstacles (goal and walls)
        for obs in self._obs:
            if obs.reward < 0:  # Pit
                color = 'r'
            else:  # Goal
                color = 'g'
            circle = plt.Circle(obs.pos, obs.rad, color=color, alpha=0.5)
            ax.add_artist(circle)

        # Add legend for clarity
        ax.legend()

        # Save the result
        plt.savefig('Results/gridworld-enhanced.png' if policy_algorithm is None else f"Results/gridworld-c-{policy_algorithm}.png")