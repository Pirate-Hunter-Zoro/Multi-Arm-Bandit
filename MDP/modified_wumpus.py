#!/usr/bin/env python3

import random
from MDP.wumpus import WumpusMDP, WumpusState

class ModifiedWumpusState(WumpusState):
    def __init__(self, x, y, has_gold, has_immunity, width, height):
        super().__init__(x, y, has_gold, has_immunity, width, height)
        # Different pits have different probabilities of killing the player
        # Step cost is either c1 or c2, c1 > c2, depending on whether the player is stepping into a location next to a pit/wumpus or not respectively
        # Reward for picking up gold is G, which you lose if you drop the gold, and is W for returning to starting location with gold, where W >> G
        self._drop_prob = 0.1

    def clone(self):
        return ModifiedWumpusState(self.x, self.y, self.has_gold, self.has_immunity, self._width, self._height)

    def next_to_pit(self, obs: dict[str, dict[tuple[int,int], float]]) -> bool:
        if 'pit' in obs.keys():
            # Look left, right, up, and down
            left_tuple = (self.x - 1, self.y)
            right_tuple = (self.x + 1, self.y)
            up_tuple = (self.x, self.y + 1)
            down_tuple = (self.x, self.y - 1)
            if (left_tuple in obs['pit']) or (right_tuple in obs['pit']) or (up_tuple in obs['pit']) or (down_tuple in obs['pit']):
                return True
        return False

    def next_to_wumpus(self, obs: dict[str, dict[tuple[int,int], float]]) -> bool:
        if 'wumpus' in obs.keys():
            # Look left, right, up, and down
            left_tuple = (self.x - 1, self.y)
            right_tuple = (self.x + 1, self.y)
            up_tuple = (self.x, self.y + 1)
            down_tuple = (self.x, self.y - 1)
            if (left_tuple in obs['wumpus']) or (right_tuple in obs['wumpus']) or (up_tuple in obs['wumpus']) or (down_tuple in obs['wumpus']):
                return True
        return False

    def __repr__(self):
        return 'pos: ({x}, {y}) has_gold: {has_gold} has_immunity: {has_immunity}'.format(**self.__dict__)

class ModifiedWumpusMDP(WumpusMDP):
    """A modified version of the Wumpus World:
        - Different pits have different probabilities of killing the player
        - Step cost is either c1 or c2, c1 > c2, depending on whether the player is stepping into a location next to a pit/wumpus or not respectively
        - Reward for picking up gold is G, which you lose if you drop the gold, and is W for returning to starting location with gold, where W >> G
    """ 
    def __init__(self, w, h, move_cost=-0.1, gold_reward=10):
        super().__init__(w, h, move_cost, gold_reward)
        self.pit_death_probs = {}
        self.adj_obs_living_cost = 2 * self.move_cost

    def add_obstacle(self, kind, pos, reward=None):
        # Handle parent obstacle addition, but we also need to 
        super().add_obstacle(kind, pos, reward)
        if kind == 'pit':
            # Assign a random (from a normal distribution) probability of dying to this pit and keep track of it
            self.pit_death_probs[tuple(pos)] = random.normalvariate(0, 1)

    def is_terminal(self, state):
        if super().is_terminal(state):
            # Checks for wumpus and goal already
            return True
        else:
            # If we are in a pit, there is a chance we die
            if self.obs_at('pit', state.pos):
                # Each pit has a different probability of killing the player - which we stored when the pit was added
                return random.uniform(0,1) < self.pit_death_probs[tuple(state.pos)]

    def r(self, s1, s2):
        ## if s1 is terminal, return whatever reward would have been associated with it
        if self.is_terminal(s1):
            return self.r(s1, s1)

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

        ## if the player has gold and is dropping it, return the negative gold reward
        if s1.has_gold and not s2.has_gold:
            return -self.gold_reward

        ## if the player is next to a pit/wumpus, return the higher step cost
        if s2.next_to_pit(self._obs) or s2.next_to_wumpus(self._obs):
            return self.adj_obs_living_cost

        ## otherwise return the move cost
        return self.move_cost
    
    def move(self, state, action):
        state, probs = super().move(state, action)
        # If the player has gold, there is a chance they will drop it
        if state.has_gold and random.uniform(0,1) < self._drop_prob:
            state.has_gold = False
        return state, probs