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
    def __init__(self, w, h, move_cost=-0.1):
        self.width = w
        self.height = h
        self.move_cost = move_cost
        self.goal_x = None
        self.goal_y = None
        self.walls = []
        self._obs = []

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
        return np.array([0,0])

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

    def feature_vector(self, state, action):
        """
        Returns the feature vector for a given state and action.
        The feature vector is a list of features that represent the state-action pair.
        """
        # Example feature vector: [x, y, action]
        assert self.goal_x is not None and self.goal_y is not None, "Goal position must be set before calculating feature vector."
        goal_distance = self.normalize_distance(state)
        action_vec = _ACTION_VECS[action]
        return np.array([state[0], state[1], *action_vec, state[0]/self.width, state[1]/self.height, goal_distance])

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