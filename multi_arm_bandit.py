from BanditSimulator.bandit_sim import Bandit_Sim
from Algorithms.ucb import ucb_algorithm, epsilon_greedy_algorithm, random_algorithm, plot_results

def compare_algorithms(bs, simulator_name=None):
    individual_ucb, cumulative_ucb, chosen_ucb = ucb_algorithm(bs, num_episodes=1000)
    individual_random, cumulative_random, chosen_random = random_algorithm(bs, num_episodes=1000)
    individual_epsilon_01, cumulative_epsilon_01, chosen_epsilon_01 = epsilon_greedy_algorithm(bs, num_episodes=1000, epsilon=0.1)
    individual_epsilon_02, cumulative_epsilon_02, chosen_epsilon_02 = epsilon_greedy_algorithm(bs, num_episodes=1000, epsilon=0.2)
    individual_epsilon_03, cumulative_epsilon_03, chosen_epsilon_03 = epsilon_greedy_algorithm(bs, num_episodes=1000, epsilon=0.3)

    plot_results(individual_random, cumulative_random, chosen_random, individual_ucb, cumulative_ucb, chosen_ucb, first_method="Random", second_method="UCB" if simulator_name==None else f"UCB_{simulator_name}")
    plot_results(individual_epsilon_01, cumulative_epsilon_01, chosen_epsilon_01, individual_ucb, cumulative_ucb, chosen_ucb, first_method="Epsilon 0.1", second_method="UCB" if simulator_name==None else f"UCB_{simulator_name}")
    plot_results(individual_epsilon_02, cumulative_epsilon_02, chosen_epsilon_02, individual_ucb, cumulative_ucb, chosen_ucb, first_method="Epsilon 0.2", second_method="UCB" if simulator_name==None else f"UCB_{simulator_name}")
    plot_results(individual_epsilon_03, cumulative_epsilon_03, chosen_epsilon_03, individual_ucb, cumulative_ucb, chosen_ucb, first_method="Epsilon 0.3", second_method="UCB" if simulator_name==None else f"UCB_{simulator_name}")

bs = Bandit_Sim(n_arms=5, payout_std=0.1, seed=42)
compare_algorithms(bs, "5_arms_0.1_std")

# Now let's do the same but when the simulator has more arms and a higher standard deviation
bs = Bandit_Sim(n_arms=50, payout_std=0.5, seed=42)
compare_algorithms(bs, "50_arms_0.5_std")