from MDP import gridworld
from MDP.wumpus import WumpusMDP

# The navigation problem introduced in Section 17.1
first_env = gridworld.GridWorld(3, 4, -0.04)
first_env.add_obstacle('wall', (1,1))
first_env.add_obstacle('pit', (1,2), -1)
first_env.add_obstacle('goal', (2,3), 1)

# A modified version of the Wumpus World:
# Different pits have different probabilities of killing the player
# Step cost is either c1 or c2, c1 > c2, depending on whether the player is stepping into a location next to a pit/wumpus or not respectively
# Reward for picking up gold is G, which you lose if you drop the gold, and is W for returning to starting location with gold, where W >> G
second_env = WumpusMDP(10, 10, -0.04)