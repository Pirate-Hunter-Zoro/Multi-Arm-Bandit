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
    V = {s.i: float('-inf') if not env.is_terminal(s) else 0 for s in env.states}
    # Random action for each state
    policy = {}
    for s in env.states:
        actions = env.actions_at(s)
        assert len(actions) > 0 or env.is_terminal(s), "No actions available for non-terminal state {}".format(s)
        if not env.is_terminal(s):
            policy[s.i] = actions[int(random.random() * len(actions))]

    num_eval_steps = 10  # Number of evaluation steps for each policy iteration

    while True:
        old_policy = policy.copy()
        
        # Policy evaluation step
        for _ in range(num_eval_steps):
            V_copy = V.copy()
            for s in env.states:
                if not env.is_terminal(s):
                    # Evaluate our policy
                    action = policy[s.i]
                    next_states_with_probs = env.p(s, action)
                    value = 0
                    for next_state, prob in next_states_with_probs:
                        value += prob * (env.r(s, next_state) + gamma * (V_copy[next_state.i] if V_copy[next_state.i] != float('-inf') else 0))
                    V[s.i] = value
                    
        # Policy improvement step
        for s in env.states:
            if not env.is_terminal(s):
                # Choose the action that maximizes the expected value function
                actions = env.actions_at(s)
                assert len(actions) > 0, "No actions available for state {}".format(s)
                best_action = actions[int(random.random() * len(actions))]
                best_value = float('-inf')
                for action in actions:
                    value = 0
                    for s_prime, prob in env.p(s, action):
                        value += prob * (env.r(s, s_prime) + gamma * V[s_prime.i])
                    if value > best_value:
                        best_value = value
                        best_action = action
                policy[s.i] = best_action

        # See of the policy changed
        if old_policy == policy:
            break

    return policy