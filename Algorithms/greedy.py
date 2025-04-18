from random import random

import numpy as np


def safe_greedy_action(env, current_state):
    # List of preferred directions toward the goal
    directions = env.actions_at(current_state)
    best_action = None
    min_dist_to_goal = float('inf')

    for action in directions:
        next_state, _ = env.act(current_state, action)
        # Check for collisions with obstacles
        is_safe = all(
            np.linalg.norm(np.array(next_state) - np.array(obs.pos)) > obs.rad
            for obs in env._obs if obs.reward < 0
        )
        if is_safe:
            dist_to_goal = np.linalg.norm(np.array(next_state) - np.array([env.goal_x, env.goal_y]))
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
                best_action = action

    return best_action if best_action else random.choice(env.actions)


def run_greedy(env, env_name=''):
    state = env.initial_state
    path = [state]
    done = False

    while not done:
        action = safe_greedy_action(env, state)
        if action is None:
            print("No safe action available! Trapped?")
            break

        state, _ = env.act(state, action)
        path.append(state)

        if env.is_terminal(state):
            done = True

    env.display(path, policy_algorithm="greedy" if env_name is None else f"{env_name}_greedy")