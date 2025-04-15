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
    PICK_UP=5


_TXT = {
    Actions.UP: "^^",
    Actions.DOWN: "vv",
    Actions.LEFT: "<<",
    Actions.RIGHT: ">>",
    Actions.PICK_UP: "[]"
}


_UP = np.array([0, 1])
_DOWN = np.array([0, -1])
_LEFT = np.array([-1, 0])
_RIGHT = np.array([1, 0])


class WumpusState(MDPState):
    def __init__(self, x, y, has_gold, has_immunity, width, height):
        self.x = x
        self.y = y
        self.has_gold = has_gold
        self.has_immunity = has_immunity
        self._width = width
        self._height = height

    def clone(self):
        return WumpusState(self.x, self.y, self.has_gold, self.has_immunity, self._width, self._height)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def i(self):
        g = 2 if self.has_gold else 0
        i = 1 if self.has_immunity else 0
        return (self._height * self.x + self.y) * (g+i)

    def __repr__(self):
        return 'pos: ({x}, {y}) has_gold: {has_gold} has_immunity: {has_immunity}'.format(**self.__dict__)


def _clip(p, max_x, max_y):
    p = np.array([max(min(p[0], max_x), 1),
                  max(min(p[1], max_y), 1)])
    return p


_OBS_KEYS = ['wumpus', 'pit', 'goal']
_OBS_REWARDS = {
    'wumpus': -5.0,
    'pit': -1.0,
    'goal': 10.0
}
_OBJ_KEYS = ['gold', 'immune']
class WumpusMDP(FiniteStateMDP):
    def __init__(self, w, h, move_cost=-0.1, gold_reward=10):
        self._w = w
        self._h = h
        self.move_cost = move_cost
        self.gold_reward = gold_reward
        self._obs = {k:{} for k in _OBS_KEYS}
        self._obj = {k:set([]) for k in _OBJ_KEYS}

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
        return [WumpusState(*x, self.width, self.height) for x in
                    itertools.product(range(self.width), range(self.height),
                                      (True, False), (True, False))]

    @property
    def actions(self):
        return Actions

    @property
    def initial_state(self):
        return WumpusState(0, 0, False, False, self.width, self.height)

    def actions_at(self, state):
        a = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
        if self.obj_at('gold', state.pos) or self.obj_at('immune', state.pos):
            a += [Actions.PICK_UP]
        return a

    def p(self, state, action):
        if action in [Actions.PICK_UP]:
            return self.pick_up(state, action)
        elif action in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]:
            return self.move(state, action)
        else:
            raise Exception("Invalid action specified: {}".format(action))

    def r(self, s1, s2):
        ## if it's the pit, return the default pit bad reward
        if self.obs_at('pit', s2.pos):
            return self._obs['pit'][tuple(s2.pos)]

        ## if it's the goal, return the goal reward + gold reward if applicable
        if self.obs_at('goal', s2.pos):
            gr = self.gold_reward if s2.has_gold else 0
            return self._obs['goal'][tuple(s2.pos)] + gr

        ## if it's the wumpus, return the negative wumpus reward unless player has immunity
        if self.obs_at('wumpus', s2.pos) and not s2.has_immunity:
            return self._obs['wumpus'][tuple(s2.pos)]

        ## otherwise return the move cost
        return self.move_cost

    def is_terminal(self, state):
        ## if we're at the wumpus and have no immunity, we die
        if self.obs_at('wumpus', state.pos) and not state.has_immunity:
            return True

        ## if we're at the goal, we win
        if self.obs_at('goal', state.pos):
            return True

        return False

    def obs_at(self, kind, pos):
        return tuple(pos) in self._obs[kind].keys()

    def obj_at(self, kind, pos):
        return tuple(pos) in self._obj[kind]

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
            new_pos = _clip(state.pos + a, self.width, self.height)
            new_state.x = new_pos[0]
            new_state.y = new_pos[1]
            x += [new_state]

        return zip(x, probs)

    def pick_up(self, state, action):
        new_state = state.clone()
        if self.obj_at('gold', state.pos):
            new_state.has_gold = True
        if self.obj_at('immune', state.pos):
            new_state.has_immunity = True
        return [(new_state, 1.0)]

    def add_obstacle(self, kind, pos, reward=None):
        ## default rewards
        if not reward:
            reward = _OBS_REWARDS[kind]
        self._obs[kind][tuple(pos)] = reward

    def add_object(self, kind, pos):
        self._obj[kind].add(tuple(pos))

    def display(self):
        obs_lab = lambda p, lab, kind: lab if self.obs_at(kind, p) else ' '
        obj_lab = lambda p, lab, kind: lab if self.obj_at(kind, p) else ' '
        print('      ', end='')
        for i in range(self.width):
            print(' {:5d} '.format(i),end='')
        print()
        for j in reversed(range(self.height)):
            print('{:5d} '.format(j), end='')
            for i in range(self.width):
                p = tuple([i, j])
                l_s = 'S' if p == (0, 0) else ' '
                l_w = obs_lab(p, 'W', 'wumpus')
                l_p = obs_lab(p, 'P', 'pit')
                l_gl = obs_lab(p, 'X', 'goal')
                l_gd = obj_lab(p, 'G', 'gold')
                l_i = obj_lab(p, 'I', 'immune')
                print('|' + l_s+l_w+l_p+l_gl+l_gd+l_i,end='')
            print('|')

    def display(self, states):
        """
        Helper method to display the gridworld after an agent has traversed a certain path through it
        """
        _, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        gold_and_immune_posns = [(state.x, state.y) for state in states if state.has_gold and state.has_immunity]
        gold_posns = [(state.x, state.y) for state in states if state.has_gold and not state.has_immunity]
        immune_posns = [(state.x, state.y) for state in states if not state.has_gold and state.has_immunity]
        trash_posns = [(state.x, state.y) for state in states if not state.has_gold and not state.has_immunity]
        ax.scatter(x=[p[0] for p in gold_and_immune_posns], y=[p[1] for p in gold_and_immune_posns], marker='x', color='green', alpha=0.5, label='Agent - G&I')
        ax.scatter(x=[p[0] for p in gold_posns], y=[p[1] for p in gold_posns], marker='x', color='yellow', alpha=0.5, label='Agent - G')
        ax.scatter(x=[p[0] for p in immune_posns], y=[p[1] for p in immune_posns], marker='x', color='blue', alpha=0.5, label='Agent - I')
        ax.scatter(x=[p[0] for p in trash_posns], y=[p[1] for p in trash_posns], marker='x', color='black', alpha=0.5, label='Agent - T')

        # Now show all of the obstacles - including walls and pits and goals
        obstacle_types = ['wall', 'goal', 'pit', 'gold', 'immune', 'wumpus']
        for obs_type in obstacle_types:
            if obs_type not in self._obs.keys():
                self._obs[obs_type] = {}

        wall_posns = self._obs['wall'].keys()
        goal_posns = self._obs['goal'].keys()
        pit_posns = self._obs['pit'].keys()
        gold_posns = self._obj['gold']
        immune_posns = self._obj['immune']
        wumpus_posns = self._obs['wumpus'].keys()

        ax.scatter(x=[p[0] for p in wall_posns], y=[p[1] for p in wall_posns], color='indigo', alpha=0.5, label='Wall')
        ax.scatter(x=[p[0] for p in goal_posns], y=[p[1] for p in goal_posns], color='orange', alpha=0.5, label='Goal')
        ax.scatter(x=[p[0] for p in pit_posns], y=[p[1] for p in pit_posns], color='gray', alpha=0.5, label='Pit')
        ax.scatter(x=[p[0] for p in gold_posns], y=[p[1] for p in gold_posns], color='gold', alpha=0.5, label='Gold')
        ax.scatter(x=[p[0] for p in immune_posns], y=[p[1] for p in immune_posns], color='cyan', alpha=0.5, label='Immunity')
        ax.scatter(x=[p[0] for p in wumpus_posns], y=[p[1] for p in wumpus_posns], color='purple', alpha=0.5, label='Wumpus')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        ax.set_title('Wumpus World')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        plt.savefig('wumpus.png')