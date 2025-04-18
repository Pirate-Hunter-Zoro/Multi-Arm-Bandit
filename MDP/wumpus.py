#!/usr/bin/env python3

from enum import Enum
import math

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


def cantor_pair(x, y):
    return (x + y) * (x + y + 1) // 2 + y

def unique_int(a, b, c, d):
    return cantor_pair(
            cantor_pair(
                cantor_pair(a, b),
                c
            ),
            d
        )

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
        g = 2 if self.has_gold else 1
        i = 1 if self.has_immunity else 0
        return unique_int(self.x, self.y, g, i)

    def __repr__(self):
        return 'pos: ({x}, {y}) has_gold: {has_gold} has_immunity: {has_immunity}'.format(**self.__dict__)


def _clip(p, max_x, max_y):
    p = np.array([max(min(p[0], max_x), 0),
                  max(min(p[1], max_y), 0)])
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
        a = []
        if not self.is_terminal(state):
            a = [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
            if (self.obj_at('gold', state.pos) and not state.has_gold) or (self.obj_at('immune', state.pos) and not state.has_immunity):
                ## if we can pick up gold or immunity, add that action
                a += [Actions.PICK_UP]
        return a

    def p(self, state, action):
        assert not self.is_terminal(state), "Cannot take action in terminal state"
        if action in [Actions.PICK_UP]:
            return self.pick_up(state)
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
        assert action in [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT], "Invalid action specified: {}".format(action)
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
            new_state.x = new_pos[0]
            new_state.y = new_pos[1]
            x += [new_state]

        return zip(x, probs)

    def pick_up(self, state):
        new_state = state.clone()
        assert self.obj_at('gold', state.pos) or self.obj_at('immune', state.pos), "Cannot pick up object that doesn't exist"
        if self.obj_at('gold', state.pos):
            assert not state.has_gold, "Cannot pick up gold when already holding it"
            new_state.has_gold = True
        if self.obj_at('immune', state.pos):
            assert not state.has_immunity, "Cannot pick up immunity when already holding it"
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

    def display(self, states, policy_algorithm=None):
        """
        Helper method to display the gridworld after an agent has traversed a certain path through it
        """
        cols = 1
        rows = math.ceil(len(states) / cols)
        
        plt.grid(True)
        fig, axes = plt.subplots(rows, cols, figsize=(self.width * cols, self.height * rows), dpi=100)
        axes = axes.flatten()  # Flatten to 1D for easy indexing
        
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
        posns = [(state.x, state.y) for state in states]

        # If only one plot, axes won't be a list
        if len(states) == 1:
            axes = [axes]

        for i in range(len(states)):
            axes[i].set_xlim(0, self.width)
            axes[i].set_ylim(0, self.height)
            axes[i].set_aspect('equal')
            
            axes[i].set_title(f"Step {i}")
            agent_color = "black"
            agent_label = "Agent - G&I" if (states[i].has_gold and states[i].has_immunity) else ("Agent - I" if states[i].has_immunity else ("Agent - G" if states[i].has_gold else "Agent - T"))
            axes[i].scatter(x=[posns[i][0]], y=[posns[i][1]], marker='x', color=agent_color, alpha=0.5, label=agent_label)
            axes[i].scatter(x=[p[0] for p in wall_posns], y=[p[1] for p in wall_posns], color='indigo', alpha=0.5, label='Wall')
            axes[i].scatter(x=[p[0] for p in goal_posns], y=[p[1] for p in goal_posns], color='orange', alpha=0.5, label='Goal')
            axes[i].scatter(x=[p[0] for p in pit_posns], y=[p[1] for p in pit_posns], color='gray', alpha=0.5, label='Pit')
            axes[i].scatter(x=[p[0] for p in gold_posns], y=[p[1] for p in gold_posns], color='gold', alpha=0.5, label='Gold')
            axes[i].scatter(x=[p[0] for p in immune_posns], y=[p[1] for p in immune_posns], color='cyan', alpha=0.5, label='Immunity')
            axes[i].scatter(x=[p[0] for p in wumpus_posns], y=[p[1] for p in wumpus_posns], color='purple', alpha=0.5, label='Wumpus')
            
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axes[i].set_xlabel('Column')
            axes[i].set_ylabel('Row')
           
        fig.tight_layout()
        fig.savefig('Results/wumpus.png' if policy_algorithm is None else f'Results/wumpus-{policy_algorithm}.png')
        plt.close(fig)