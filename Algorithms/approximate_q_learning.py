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
        q_values = [np.dot(self.theta, f) for f in features]
        best_action = self.env.actions[np.argmax(q_values)]
        return best_action

def approximate_q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=1):
    """
    Performs approximate Q-learning with linear function approximation.
    """
    # Initialize weights for function approximation
    theta = np.zeros(len(env.feature_vector(env.initial_state, list(env.actions)[0])))  # Initialize theta to zeros

    epsilon_decay = 0.99

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
            max_q_next = max(next_q_values, default=0)

            # Compute TD error
            td_error = reward + gamma * max_q_next - np.dot(theta, feature_vec)

            # Update weights using gradient descent
            theta += alpha * td_error * feature_vec

            state = next_state
            done = env.is_terminal(state)

        # Decay epsilon to encourage exploitation
        epsilon = max(0.1, epsilon * epsilon_decay)

    return LinearQTablePolicy(theta, env)

def run_approximate_q_learning(env, num_episodes=10000, env_name=''):
    x = env.initial_state
    states = [x]
    policy = approximate_q_learning(env)
    for _ in range(num_episodes):
        x, _ = env.act(x, policy(x))
        states.append(x)
        if env.is_terminal(x):
            break
    env.display(states, policy_algorithm=f'{env_name}_approximate_q_learning')