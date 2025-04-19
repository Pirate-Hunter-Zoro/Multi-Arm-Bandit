from Algorithms.approximate_q_learning import run_approximate_q_learning
from Algorithms.q_learning import run_q_learning
from MDP.gridworld_c import ContinuousGridWorldMDP

# The 4x3 world described in the Chapter 17.
first_env = ContinuousGridWorldMDP(3, 4, -0.04)
first_env.add_goal(2, 3, radius=1)
first_env.add_pit(2, 2, radius=1)
first_env.add_wall(1, 1, radius=1)

# A 10x10 world variant with no obstacles and a +1 reward at (10,10).
second_env = ContinuousGridWorldMDP(10, 10, -0.04)
second_env.add_goal(9, 9, radius=1)

# A 10x10 world variant with no obstacles and a +1 reward at (5,5).
third_env = ContinuousGridWorldMDP(10, 10, -0.04)
third_env.add_goal(4, 4, radius=1)


if __name__ == '__main__':
    run_q_learning(first_env, env_name='first_env')
    run_q_learning(second_env, env_name='second_env')
    run_q_learning(third_env, env_name='third_env')
    run_approximate_q_learning(first_env, env_name='first_env')
    run_approximate_q_learning(second_env, env_name='second_env')
    run_approximate_q_learning(third_env, env_name='third_env')