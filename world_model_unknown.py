from MDP import gridworld

# The 4x3 world described in the Chapter 17.
first_env = gridworld.GridWorld(3, 4, -0.04)
first_env.add_obstacle('wall', (1,1))
first_env.add_obstacle('pit', (1,2), -1)
first_env.add_obstacle('goal', (2,3), 1)

# A 10x10 world variant with no obstacles and a +1 reward at (10,10).
second_env = gridworld.GridWorld(10, 10, -0.04)
second_env.add_obstacle('goal', (9, 9), 1)

# A 10x10 world variant with no obstacles and a +1 reward at (5,5).
third_env = gridworld.GridWorld(10, 10, -0.04)
third_env.add_obstacle('goal', (4, 4), 1)

