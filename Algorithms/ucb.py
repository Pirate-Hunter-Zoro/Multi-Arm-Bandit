import math
import random
from Algorithms.heap import Heap

# We'll write an algorithm to solve the multi-armed bandit problem using UCB (Upper Confidence Bound) algorithm.

def ucb_algorithm(bandit_sim, num_episodes=1000):
    """
    UCB algorithm for the multi-armed bandit problem.
    
    Parameters:
    bandit_sim : Bandit_Sim
        The bandit simulator object.
    num_episodes : int
        The number of episodes to run the algorithm.
    """
    # Keep track of the following heuristic for each arm: 
    # (mean reward) + (exploration bonus)
    heuristic = lambda num_pulls, gained_from_arm, total_pulls: gained_from_arm / num_pulls + math.sqrt(2 * math.log(total_pulls) / num_pulls)
    payouts = []
    arm_pull_counts = []
    sequential_payoffs = []
    sequential_individual_payoffs = []
    chosen_arms = []
    total_pulls = 0
    total_reward = 0

    # Initialize the number of pulls and successes for each arm by pulling each one once
    for i in range(bandit_sim.n_arms):
        payouts.append(bandit_sim.pull_arm(i))
        total_reward += payouts[i]
        sequential_payoffs.append(total_reward)
        sequential_individual_payoffs.append(payouts[i])
        arm_pull_counts.append(1)
        total_pulls += 1
        chosen_arms.append(i)
    
    arm_heap = Heap(lambda arm_1, arm_2: heuristic(arm_pull_counts[arm_1], payouts[arm_1], total_pulls) > heuristic(arm_pull_counts[arm_2], payouts[arm_2], total_pulls))
    arms = [i for i in range(bandit_sim.n_arms)]
    arm_heap.heapify(arms)
    # Now we can start the algorithm
    for _ in range(num_episodes):
        # Pull the arm with the highest heuristic and update values accordingly
        arm = arm_heap.pop()
        arm_heap.clear()
        payout = bandit_sim.pull_arm(arm)
        payouts[arm] += payout
        total_reward += payout
        sequential_individual_payoffs.append(payout)
        arm_pull_counts[arm] += 1
        total_pulls += 1
        # Rebuild the heap with the updated values
        arms = [i for i in range(bandit_sim.n_arms)]
        arm_heap.heapify(arms)
        # Store the sequential payoffs
        sequential_payoffs.append(total_reward)
        chosen_arms.append(arm)

    return sequential_individual_payoffs, sequential_payoffs, chosen_arms


def epsilon_greedy_algorithm(bandit_sim, num_episodes=1000, epsilon=0.1):
    """
    Epsilon-greedy algorithm for the multi-armed bandit problem.
    
    Parameters:
    bandit_sim : Bandit_Sim
        The bandit simulator object.
    num_episodes : int
        The number of episodes to run the algorithm.
    epsilon : float
        The probability of exploring a random arm instead of exploiting the best arm.
    """
    # Keep track of which arm has the highest payout mean
    highest_mean_payout = float('-inf')
    best_arm = -1
    individual_payoffs = []
    cumulative_payoffs = []
    chosen_arms = []
    total_reward = 0
    arm_rewards = [0] * bandit_sim.n_arms
    arm_counts = [0] * bandit_sim.n_arms
    # Initialize the number of pulls and successes for each arm by pulling each one once
    for i in range(bandit_sim.n_arms):
        payout = bandit_sim.pull_arm(i)
        arm_rewards[i] += payout
        arm_counts[i] += 1
        total_reward += payout
        individual_payoffs.append(payout)
        cumulative_payoffs.append(total_reward)
        # Update the best arm if necessary
        if arm_rewards[i] / arm_counts[i] > highest_mean_payout:
            highest_mean_payout = arm_rewards[i] / arm_counts[i]
            best_arm = i
        chosen_arms.append(i)

    # Now we can start the algorithm
    for _ in range(num_episodes):
        # Decide whether to explore or exploit
        if random.random() < epsilon:
            # Pick a random arm
            arm = int(random.random() * bandit_sim.n_arms)
        else:
            # Pick the best arm
            arm = best_arm

        # Pull the arm and update values accordingly
        payout = bandit_sim.pull_arm(arm)
        arm_rewards[arm] += payout
        arm_counts[arm] += 1
        total_reward += payout
        individual_payoffs.append(payout)
        cumulative_payoffs.append(total_reward)

        # Update the best arm if necessary
        if arm_rewards[arm] / arm_counts[arm] > highest_mean_payout:
            highest_mean_payout = arm_rewards[arm] / arm_counts[arm]
            best_arm = arm
        
        chosen_arms.append(arm)

    return individual_payoffs, cumulative_payoffs, chosen_arms


def random_algorithm(bandit_sim, num_episodes=1000):
    """
    Random pulling for the multi-armed bandit problem.
    
    Parameters:
    bandit_sim : Bandit_Sim
        The bandit simulator object.
    num_episodes : int
        The number of episodes to run the algorithm.
    """
    # Keep track of the following heuristic for each arm: 
    # (mean reward) + (exploration bonus)
    payouts = []
    sequential_payoffs = []
    sequential_individual_payoffs = []
    chosen_arms = []
    total_reward = 0

    for i in range(bandit_sim.n_arms):
        # Start with pulling each arm once
        payouts.append(bandit_sim.pull_arm(i))
        total_reward += payouts[i]
        sequential_payoffs.append(total_reward)
        sequential_individual_payoffs.append(payouts[i])
        chosen_arms.append(i)

    # Now we can start the algorithm
    for _ in range(num_episodes):
        # Pull the arm with the highest heuristic and update values accordingly
        arm = int(random.random() * bandit_sim.n_arms)
        payout = bandit_sim.pull_arm(arm)
        payouts[arm] += payout
        total_reward += payout
        sequential_individual_payoffs.append(payout)
        # Store the sequential payoffs
        sequential_payoffs.append(total_reward)
        chosen_arms.append(arm)

    return sequential_individual_payoffs, sequential_payoffs, chosen_arms


def plot_results(first_individual, first_cumulative, first_chosen, second_individual, second_cumulative, second_chosen, first_method, second_method):
    """
    Plot the results of the UCB and random algorithms.
    """
    import matplotlib.pyplot as plt

    time_steps = range(len(first_individual))

    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

    # ---- Top Plot: Individual Payoffs ----
    axes[0].plot(time_steps, first_individual, label=first_method, color='blue')
    axes[0].plot(time_steps, second_individual, label=second_method, color='orange', linestyle='--')
    axes[0].set_ylabel('Individual Payoff')
    axes[0].set_title('Individual Pull Payoff Over Time')
    axes[0].legend()
    axes[0].grid(True)

    # ---- Middle Plot: Cumulative Payoffs ----
    axes[1].plot(time_steps, first_cumulative, label=first_method, color='blue')
    axes[1].plot(time_steps, second_cumulative, label=second_method, color='orange', linestyle='--')
    axes[1].set_ylabel('Cumulative Payoff')
    axes[1].set_title('Cumulative Payoff Over Time')
    axes[1].legend()
    axes[1].grid(True)

    # ---- Bottom Plot: Chosen Arms ----
    axes[2].scatter(time_steps, first_chosen, label=first_method, color='blue', marker='o')
    axes[2].scatter(time_steps, second_chosen, label=second_method, color='orange', marker='x')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Chosen Arm')
    axes[2].set_title('Chosen Arm Over Time')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'Results/multi_arm_bandit_{first_method}_vs_{second_method}.png')