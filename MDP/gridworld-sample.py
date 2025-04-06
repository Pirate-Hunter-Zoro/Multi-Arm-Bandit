#!/usr/bin/env python3

from gridworld import DiscreteGridWorldMDP
import numpy as np


mdp = DiscreteGridWorldMDP(50, 50)

mdp.add_obstacle('pit', (5, 5))
mdp.add_obstacle('pit',(5, 10))
mdp.add_obstacle('pit', (10, 20))
mdp.add_obstacle('pit', (40, 41))

for i in range(20, 30):
    mdp.add_obstacle('wall', (i, i))
for j in range(50, 80):
    mdp.add_obstacle('wall', (20, j))

mdp.add_obstacle('goal', (50, 50))

x = mdp.initial_state
t = 0
states = []

while not mdp.is_terminal(x) and t < 1000:
    states.append(x)
    print(x)
    a = np.random.choice(list(mdp.actions_at(x)))
    print(a)
    x, _ = mdp.act(x, a)
    t += 1

mdp.display(states)