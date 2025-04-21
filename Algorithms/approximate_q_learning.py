import numpy as np
import random

class LinearQTablePolicy:
    def __init__(self, theta, env):
        self.theta = theta  # Weights for function approximation
        self.env = env
    
    def __call__(self, state):
        """
        Given a state, return the action with the highest Q-value.
        If no actions are available, return None.
        """
        features = [self.env.feature_vector(state, action) for action in self.env.actions_at(state)]
        q_values = [np.dot(self.theta, f) for f in features] # Dot product of theta and feature vector representing the state/action pair, FOR ALL such actions that can go with the state
        best_action = self.env.actions[np.argmax(q_values)] # Pick the action that yields the highest dot product
        return best_action

def approximate_q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=1):
    """
    Performs approximate Q-learning with linear function approximation.
    """
    # Initialize weights for function approximation
    theta = np.zeros(len(env.feature_vector(env.initial_state, list(env.actions)[0])))  # Initialize theta to zeros

    epsilon_decay = 0.995
    epsilon_min = 0.1

    for _ in range(num_episodes):
        state = env.initial_state
        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Pick a random action
                action = random.choice(env.actions)
            else:
                # Use the current policy to select the best action
                action = LinearQTablePolicy(theta, env)(state)

            # Take action and observe next state and reward
            next_state, reward = env.act(state, action)

            # Compute feature vector for current state-action pair
            feature_vec = env.feature_vector(state, action)

            # Compute max Q-value for next state (using greedy policy)
            next_features = [env.feature_vector(next_state, a) for a in env.actions_at(next_state)]
            next_q_values = [np.dot(theta, f) for f in next_features]
            assert all(not np.any(np.isnan(q)) and not np.any(np.isinf(q)) for q in next_q_values), f"Invalid next Q-values: {next_q_values}"
            max_q_next = max(next_q_values, default=0)

            assert not np.any(np.isnan(reward)) and not np.any(np.isinf(reward)), f"Invalid reward: {reward}"
            assert not np.any(np.isnan(max_q_next)) and not np.any(np.isinf(max_q_next)), f"Invalid max_q_next: {max_q_next}"
            q_value = np.dot(theta, feature_vec)
            assert not np.any(np.isnan(q_value)) and not np.any(np.isinf(q_value)), f"Invalid Q-value: {q_value}"

            # Compute TD error
            td_error = reward + gamma * max_q_next - np.dot(theta, feature_vec)
            td_error = np.clip(td_error, -1e10, 1e10)  # Prevent extreme TD errors

            assert not np.any(np.isnan(td_error)) and not np.any(np.isinf(td_error)), f"Invalid TD error: {td_error}"

            # Update weights using gradient descent
            theta += alpha * td_error * feature_vec
            theta = np.clip(theta, -1e10, 1e10)  # Prevent large weight values

            state = next_state
            done = env.is_terminal(state)

        # Decay epsilon to encourage exploitation
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return LinearQTablePolicy(theta, env)

def run_approximate_q_learning(env, num_episodes=1000, env_name=''):
    x = env.initial_state
    states = [x]
    policy = approximate_q_learning(env)
    for _ in range(num_episodes):
        x, _ = env.act(x, policy(x))
        states.append(x)
        if env.is_terminal(x):
            break
    env.display(states, policy_algorithm=f'{env_name}_approximate_q_learning')