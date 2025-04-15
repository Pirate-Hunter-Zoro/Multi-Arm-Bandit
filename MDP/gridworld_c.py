#!/usr/bin/env python3

from enum import Enum
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


def _clip(p, max_x, max_y):
    p = np.array([max(min(p[0], max_x), 1),
                  max(min(p[1], max_y), 1)])
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
        self._obs = []

    def add_pit(self, x, y, radius, cost=10.0):
        self._obs += [Obstacle(x, y, radius, -cost)]

    def add_goal(self, x, y, radius, reward=10.0):
        self._obs += [Obstacle(x, y, radius, reward)]

    @property
    def actions(self):
        """Return iterable of all actions."""
        return Actions

    def actions_at(self, state):
        """Return iterable of all actions at given state."""
        return Actions

    @property
    def initial_state(self):
        """Returns initial state (assumed determinstic)."""
        return np.array([2,2])

    def _in_obs(self, state):
        for obs in self._obs:
            if obs.contains(state):
                return obs
        return None
    
    def r(self, s1, s2):
        """Returns the reward for transitioning from s1 to s2. For now, assume it is deterministic."""
        obs = self._in_obs(s2)
        if obs:
            return obs.reward

        return self.move_cost

    def is_terminal(self, state):
        """Returns true if state s is terminal."""
        if self._in_obs(state):
            return True

        return False ## not great coding, but idc

    def act(self, state, action):
        """Observe a single MDP transition."""
        mean = _VEC[action]
        cov = np.eye(2) / 4 + np.abs(np.diag(mean) / 2)
        next_state = state + np.random.multivariate_normal(mean, cov)
        next_state = _clip(next_state, self.width, self.height)
        # print(mean, cov)
        return next_state, self.r(state, next_state)

    def display(self, states):
        """
        Helper method to display the gridworld after an agent has traversed a certain path through it
        """
        _, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        x = [state[0] for state in states]
        y = [state[1] for state in states]
        plt.plot(x, y, 'bo-')

        # Now show all of the obstacles
        for obs in self._obs:
            circle = plt.Circle(obs.pos, obs.rad, color='r' if obs.reward < 0 else 'g', alpha=0.5)
            ax.add_artist(circle)

        plt.savefig('gridworld-c.png')