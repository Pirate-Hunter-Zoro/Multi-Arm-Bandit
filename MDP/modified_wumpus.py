#!/usr/bin/env python3

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

    def __repr__(self):
        return 'pos: ({x}, {y}) has_gold: {has_gold} has_immunity: {has_immunity}'.format(**self.__dict__)


class ModifiedWumpusMDP(WumpusMDP):
    def __init__(self, w, h, move_cost=-0.1, gold_reward=10):
        super().__init__(w, h, move_cost, gold_reward)

    def r(self, s1, s2):
        # TODO - fix according to new rules
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