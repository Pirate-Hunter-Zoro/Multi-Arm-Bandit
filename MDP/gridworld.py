#!/usr/bin/env python3

from enum import Enum

from matplotlib import pyplot as plt
from MDP.mdp import FiniteStateMDP, MDPState
import itertools
import numpy as np


class Actions(Enum):
    UP=1
    DOWN=2
    LEFT=3
    RIGHT=4


_TXT = {
    Actions.UP: "^^",
    Actions.DOWN: "vv",
    Actions.LEFT: "<<",
    Actions.RIGHT: ">>",
}


_UP = np.array([0, 1])
_DOWN = np.array([0, -1])
_LEFT = np.array([-1, 0])
_RIGHT = np.array([1, 0])


class GridState(MDPState):
    def __init__(self, x, y, has_gold, has_immunity, width, height):
        self.x = x
        self.y = y
        self.has_gold = has_gold
        self.has_immunity = has_immunity
        self._width = width
        self._height = height

    def clone(self):
        return GridState(self.x, self.y, self.has_gold, self.has_immunity, self._width, self._height)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def i(self):
        return int(self._height * self.x + self.y)

    def __repr__(self):
        return '({x}, {y})'.format(**self.__dict__)


def _clip(p, max_x, max_y):
    p = np.array([max(min(p[0], max_x), 0),
                  max(min(p[1], max_y), 0)])
    return p


_OBS_KEYS = ['pit','wall']
_OBS_REWARDS = {
    'pit': -10.0,
    'goal': 10.0,
}

_OBJ_KEYS = []
class DiscreteGridWorldMDP(FiniteStateMDP):
    def __init__(self, w, h, move_cost=-0.1):
        self._w = w
        self._h = h
        self.move_cost = move_cost
        self._obs = {k:{} for k in _OBS_KEYS}

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    @property
    def num_states(self):
        return self.width * self.height * 4

    @property
    def states(self):
        as_tuples = itertools.product(
            range(self.width), range(self.height), (True, False), (True, False))
        as_states = [GridState(x, y, has_gold, has_immunity, self.width, self.height) for x, y, has_gold, has_immunity in as_tuples]
        return as_states

    @property
    def actions(self):
        return Actions

    @property
    def initial_state(self):
        return GridState(0, 0, False, False, self.width, self.height)

    def actions_at(self, state):
        a = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
        return a

    def p(self, state, action):
        if action in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]:
            return self.move(state, action)
        else:
            raise Exception("Invalid action specified: {}".format(action))

    def r(self, s1, s2):
        ## if it's the pit, return the default pit bad reward
        if self.obs_at('pit', s2.pos):
            return self._obs['pit'][tuple(s2.pos)]

        ## if it's the goal, return the goal reward + gold reward if applicable
        if self.obs_at('goal', s2.pos):
            return self._obs['goal'][tuple(s2.pos)]

        ## otherwise return the move cost
        return self.move_cost

    def is_terminal(self, state):
        ## if we're at the wumpus and have no immunity, we die
        if self.obs_at('pit', state.pos):
            return True

        ## if we're at the goal, we win
        if self.obs_at('goal', state.pos):
            return True

        return False

    def obs_at(self, kind, pos):
        return tuple(pos) in self._obs[kind].keys()

    def move(self, state, action):
        probs = [0.8, 0.1, 0.1]

        if action == Actions.UP:
            alst = [_UP, _LEFT, _RIGHT]
        elif action == Actions.DOWN:
            alst = [_DOWN, _RIGHT, _LEFT]
        elif action == Actions.LEFT:
            alst = [_LEFT, _UP, _DOWN]
        elif action == Actions.RIGHT:
            alst = [_RIGHT, _DOWN, _UP]

        x = []
        for a in alst:
            new_state = state.clone()
            new_pos = _clip(state.pos + a, self.width-1, self.height-1)
            if not self.obs_at('wall', new_pos):
                new_state.x = new_pos[0]
                new_state.y = new_pos[1]
            x += [new_state]

        return zip(x, probs)

    def add_obstacle(self, kind, pos, reward=None):
        ## default rewards
        if not reward:
            reward = _OBS_REWARDS[kind] if kind in _OBS_REWARDS.keys() else self.move_cost
        if kind not in self._obs.keys():
            self._obs[kind] = {}
        self._obs[kind][tuple(pos)] = reward

    def display(self):
        obs_lab = lambda p, lab, kind: lab if self.obs_at(kind, p) else ' '
        print('      ', end='')
        for i in range(self.width):
            print(' {:5d} '.format(i),end='')
        print()
        for j in reversed(range(self.height)):
            print('{:5d} '.format(j), end='')
            for i in range(self.width):
                p = tuple([i, j])
                l_s = 'S' if p == (0, 0) else ' '
                l_p = obs_lab(p, 'P', 'pit')
                l_gl = obs_lab(p, 'X', 'goal')
                print('|' + l_s+l_p+l_gl,end='')
            print('|')

    def display(self, states, policy_algorithm=None):
        """
        Helper method to display the gridworld after an agent has traversed a certain path through it
        """
        _, axes = plt.subplots(len(states), 1, figsize=(self.width, self.height*len(states)), dpi=100)

        # Grab the wall and pit positions
        # Now show all of the obstacles - including walls and pits and goals
        obstacle_types = ['wall', 'goal', 'pit']
        for obs_type in obstacle_types:
            if obs_type not in self._obs.keys():
                self._obs[obs_type] = {}

        wall_posns = self._obs['wall'].keys()
        goal_posns = self._obs['goal'].keys()
        pit_posns = self._obs['pit'].keys()
        posns = [(state.x, state.y) for state in states]

        # If only one plot, axes won't be a list
        if len(states) == 1:
            axes = [axes]

        for i in range(len(states)):
            axes[i].set_xlim(0, self.width)
            axes[i].set_ylim(0, self.height)
            axes[i].set_aspect('equal')
            
            axes[i].set_title(f"Step {i}")
            axes[i].scatter(x=[posns[i][0]], y=[posns[i][1]], marker='x', color='green', alpha=0.5, label='Agent')
            axes[i].scatter(x=[p[0] for p in wall_posns], y=[p[1] for p in wall_posns], color='indigo', alpha=0.5, label='Wall')
            axes[i].scatter(x=[p[0] for p in goal_posns], y=[p[1] for p in goal_posns], color='orange', alpha=0.5, label='Goal')
            axes[i].scatter(x=[p[0] for p in pit_posns], y=[p[1] for p in pit_posns], color='gray', alpha=0.5, label='Pit')
            
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axes[i].set_xlabel('Column')
            axes[i].set_ylabel('Row')
           
        plt.tight_layout()
        plt.savefig('Results/grid-world.png' if policy_algorithm is None else f'Results/grid-world-{policy_algorithm}.png')
        plt.close()