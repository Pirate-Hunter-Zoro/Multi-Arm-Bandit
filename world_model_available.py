from Algorithms.modified_policy_iteration import modified_policy_iteration
from Algorithms.value_iteration import value_iteration
from MDP import gridworld
from MDP.wumpus import WumpusMDP

# The navigation problem introduced in Section 17.1
first_env = gridworld.DiscreteGridWorldMDP(3, 4, -0.04)
first_env.add_obstacle('wall', (1,1))
first_env.add_obstacle('pit', (1,2), -1)
first_env.add_obstacle('goal', (2,3), 1)

# A modified version of the Wumpus World:
# Different pits have different probabilities of killing the player
# Step cost is either c1 or c2, c1 > c2, depending on whether the player is stepping into a location next to a pit/wumpus or not respectively
# Reward for picking up gold is G, which you lose if you drop the gold, and is W for returning to starting location with gold, where W >> G
second_env = WumpusMDP(4, 4, -0.04)
second_env.add_obstacle('wumpus', (1, 1), -5)
second_env.add_obstacle('pit', (1, 2), -1)
second_env.add_obstacle('pit', (2, 1), -1)
second_env.add_obstacle('wumpus', (2, 2), -5)
second_env.add_obstacle('goal', (3, 3), 10)
second_env.add_object('immune', (3, 0))
second_env.add_object('gold', (3, 1))

if __name__ == '__main__':
    # Gridworld value iteration
    x = first_env.initial_state
    t = 0
    states = [x]

    policy = value_iteration(first_env)
    for t in range(1000):
        x, _ = first_env.act(x, policy[x.i])
        states.append(x)
        if first_env.is_terminal(x):
            break

    first_env.display(states, policy_algorithm='value_iteration')


    # Wumpus world value iteration
    x = second_env.initial_state
    t = 0
    states = [x]

    policy = value_iteration(second_env)
    for t in range(1000):
        assert not second_env.is_terminal(x), "The agent has reached a terminal state"
        assert policy[x.i] is not None, "The state has no corresponding action to follow"
        x, _ = second_env.act(x, policy[x.i])
        states.append(x)
        if second_env.is_terminal(x):
            break

    second_env.display(states, policy_algorithm='value_iteration')


    # Gridworld modified policy iteration
    x = first_env.initial_state
    t = 0
    states = [x]

    policy = modified_policy_iteration(first_env)
    for t in range(1000):
        assert not first_env.is_terminal(x), "The agent has reached a terminal state"
        assert policy[x.i] is not None, "The state has no corresponding action to follow"
        x, _ = first_env.act(x, policy[x.i])
        states.append(x)
        if first_env.is_terminal(x):
            break

    first_env.display(states, policy_algorithm='modified_policy_iteration')


    # Wumpus world modified policy iteration
    x = second_env.initial_state
    t = 0
    states = [x]

    policy = modified_policy_iteration(second_env)
    for t in range(1000):
        assert not second_env.is_terminal(x), "The agent has reached a terminal state"
        assert policy[x.i] is not None, "The state has no corresponding action to follow"
        x, _ = second_env.act(x, policy[x.i])
        states.append(x)
        if second_env.is_terminal(x):
            break

    second_env.display(states, policy_algorithm='modified_policy_iteration')