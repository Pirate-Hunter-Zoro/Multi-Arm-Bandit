import sys
import os

# Add the root project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MDP.mdp import FiniteStateMDP


def value_iteration(env: FiniteStateMDP, gamma=0.9, theta=1e-6):
    """
    Perform value iteration algorithm to find the optimal policy and value function.

    Parameters:
    - env: The environment (MDP) to solve.
    - gamma: Discount factor (0 < gamma < 1).
    - theta: A small threshold to determine convergence.

    Pseudo-code:
    Initialize V(s) to zero for all states s ∈ S
    Repeat until convergence (e.g., max change < θ):
        Δ ← 0
        For each state s ∈ S:
            v ← V(s)
            V(s) ← max_a ∑_s',r [ P(s', r | s, a) * (r + γ * V(s')) ]
            Δ ← max(Δ, |v - V(s)|)

    # Derive policy π from the final value function
    For each state s ∈ S:
        π(s) ← argmax_a ∑_s',r [ P(s', r | s, a) * (r + γ * V(s')) ]
    """
    # Calling .i on a state returns a unique (hashable) index associated with it
    V = {s.i: 0 for s in env.states}
    policy = {s.i: None for s in env.states}

    while True:
        delta = 0
        for s in env.states:
            v = V[s.i]
            # What are the possible actions to take?
            actions = env.actions_at(s)
            # Update value function based on the current best action
            for action in actions:
                s_next, reward = env.act(s, action)
                # Calculate the value of taking this action and use it to update the record for V(s)
                V[s.i] = max(V[s.i], reward + gamma * V[s_next.i])  # Bellman update
            delta = max(delta, abs(v - V[s.i]))

        # If no state changed by more than theta, we can stop
        if delta < theta:
            break

    for s in env.states:
        # Choose the action that maximizes the expected value function
        actions = env.actions_at(s)
        assert len(actions)>0, "No actions available for state {}".format(s)
        best_action = None
        best_value = float('-inf')
        for action in actions:
            s_next, reward = env.act(s, action)
            value = reward + gamma * V[s_next.i]
            if value > best_value:
                best_value = value
                best_action = action
        # Update the policy for state s
        assert best_action is not None, "No best action found for state {}".format(s)
        policy[s.i] = best_action

    return policy