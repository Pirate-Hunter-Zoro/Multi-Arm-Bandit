import random


def modified_policy_iteration(env, gamma=0.9):
    """
    Perform modified policy iteration algorithm to find the optimal policy and value function.

    Parameters:
    - env: The environment (MDP) to solve.
    - gamma: Discount factor (0 < gamma < 1).
    - theta: A small threshold to determine convergence.

    Pseudo-code:
    Initialize the policy π(s) for all states s
    Initialize the value function V(s) for all states s, arbitrary but often set to zeros

    Repeat until convergence (policy doesn't change):
        # Step 1: Policy Evaluation (Modified)
        For each state s in the state space:
            # Perform a limited number of steps of evaluation (in practice this could be a few iterations)
            For i = 1 to num_eval_steps:
                V(s) = Σ P(s'|s, π(s)) * [R(s, π(s), s') + γ * V(s')]  # Bellman update for the value function

        # Step 2: Policy Improvement
        For each state s in the state space:
            # Choose the action that maximizes the expected value function
            best_action = argmax_a Σ P(s'|s, a) * [R(s, a, s') + γ * V(s')]
            If best_action != π(s):  # If the policy changes
                π(s) = best_action  # Update policy for state s

    Return the learned policy π and value function V
    """
    # Initialize value function and policy
    V = {s.i(): 0 for s in env.states}
    # Random action for each state
    policy = {s.i(): env.actions_at(s)[int(random.random()*len(env.actions_at(s)))] for s in env.states}
    num_eval_steps = 10  # Number of evaluation steps for each policy

    while True:
        old_policy = policy.copy()

        for s in env.states:
            average_reward = 0
            for _ in range(num_eval_steps):
                # Evaluate our policy
                action = policy[s]
                s_next, reward = env.act(s, action)
                average_reward += reward
            average_reward /= num_eval_steps
            V[s] = average_reward + gamma * V[s_next]

        # Policy improvement step
        for s in env.states:
            # Choose the action that maximizes the expected value function
            actions = env.actions_at(s)
            assert len(actions) > 0, "No actions available for state {}".format(s)
            best_action = None
            best_value = float('-inf')
            for action in actions:
                s_next, reward = env.act(s, action)
                value = reward + gamma * V[s_next]
                if value > best_value:
                    best_value = value
                    best_action = action
            # Update the policy for state s
            assert best_action is not None, "No best action found for state {}".format(s)
            policy[s] = best_action

        # See of the policy changed
        if old_policy == policy:
            break

    return policy, V