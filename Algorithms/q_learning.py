from collections import defaultdict
import random
from matplotlib import pyplot as plt

class QTablePolicy:
    def __init__(self, q_table, env):
        self.q_table = q_table
        self.env = env
    
    def __call__(self, state):
        """
        Given a state, return the action with the highest Q-value.
        If no actions are available, return None.
        """
        s_disc = self.env.discretize(state)
        if self.q_table[s_disc]:
            action = max(self.q_table[s_disc], key=self.q_table[s_disc].get)
        else:
            action = random.choice(self.env.actions)
        return action

def q_learning(env, num_episodes=1000, alpha=1, gamma=0.9, epsilon=1):
    """
    Initialize Q(s, a) arbitrarily for all states s and actions a
    Set hyperparameters: learning rate α ∈ (0,1], discount factor γ ∈ [0,1], and exploration rate ε ∈ [0,1]

    For each episode:
        Initialize state s

        Repeat until s is terminal:
            With probability ε:
                Choose a random action a
            Otherwise:
                Choose action a = argmax_a' Q(s, a')

            Take action a, observe reward r and next state s'
            
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

            s ← s'

    (Optional) Decay ε over time
    """
    q_table = defaultdict(lambda: defaultdict(float))  # Q(s, a) initialized to 0 for all s, a
    epsilon_decay = 0.99
    alpha_decay = 0.99
    for _ in range(num_episodes):
        s = env.initial_state
        done = False
        while not done:
            # Epsilon-greedy action selection
            s_discretized = env.discretize(s)
            # Ensures s' is in the table
            _ = q_table[s_discretized]  # Ensure current state is in the table
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                action = max(q_table[s_discretized], key=q_table[s_discretized].get, default=random.choice(env.actions))

            s_next, reward = env.act(s, action)
            s_next_discretized = env.discretize(s_next)
            _ = q_table[s_next_discretized]  # Ensure next state is in the table
            q_table[s_discretized][action] += alpha * (reward + gamma * max(q_table[s_next_discretized].values(), default=0) - q_table[s_discretized][action])
            s = s_next
            done = env.is_terminal(s)

        # Decay alpha and epsilon, but keep them above a threshold
        epsilon = max(0.1, epsilon * epsilon_decay)  # Decay epsilon, but keep it above a threshold
        alpha = max(0.1, alpha * alpha_decay)

    return QTablePolicy(q_table, env)


def plot_value_function(q_table, env_name=''):
    """
    Visualizes the value function (max Q for each state) as a heatmap.
    """
    values = {}
    for state in q_table:
        values[state] = max(q_table[state].values(), default=0)

    xs = [s[0] for s in values]
    ys = [s[1] for s in values]
    vals = [values[s] for s in values]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xs, ys, c=vals, cmap='viridis', s=200, edgecolors='k')
    plt.colorbar(sc, label='Max Q-value')
    plt.title("Q-value Heatmap (Max Q per State)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig("Results/" + env_name + "_q_value_heat_map.png")


def run_q_learning(env, num_episodes=1000, env_name=''):
    x = env.initial_state
    states = [x]
    policy = q_learning(env)
    for _ in range(num_episodes):
        x, _ = env.act(x, policy(x))
        states.append(x)
        if env.is_terminal(x):
            break
    plot_value_function(policy.q_table, env_name=env_name)
    env.display(states, policy_algorithm=f'{env_name}_q_learning')