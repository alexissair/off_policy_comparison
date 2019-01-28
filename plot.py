import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse

from agent import Agent


ALGORITHMS_TO_EVALUATE = ['tb', 'is', 'q_lambda', 'retrace', 'q_learning']

def parse_arguments():
    parser = argparse.ArgumentParser(description='plot results for the off-policy training of an agent')
    parser.add_argument('--env', type=str, help='choose a gym environement')
    parser.add_argument('--alpha', type=float, help='the learning rate to use (default: 0.01)')
    parser.add_argument('--epsilon', type=float, help='the initial value of the greediness factor (default: 0.5)')
    parser.add_argument('--lbda', type=float, help='the lambda factor in some algorithms (default: 1)')
    parser.add_argument('--n', type=int, help='the "memory" of the off-policy algorithm (default: 100)')
    parser.add_argument('--gamma', type=float, help='the discount factor (default: 0.8)')
    parser.add_argument('--evaluate_every', type=int, help='the frequency where we evaluate the agent during training (default: 5000)')
    parser.add_argument('--episodes_to_evaluate', type=int, help='the number of episodes to use to evaluate the agent (default: 200)')
    parser.add_argument('--episodes_to_train', type=int, help='the number of episodes to train the agent (default: 10000)')
    args = parser.parse_args()

    return args

def train_and_evaluate_agents(args):
    agent_dict = {}
    dict_args  = {k:v for k,v in vars(args).items() if v is not None}
    print(args)
    for algorithm in ALGORITHMS_TO_EVALUATE:
        a_temp = Agent(algorithm_used=algorithm, **dict_args)
        if args.episodes_to_train:
            a_temp.run_multiple_episode(args.episodes_to_train)
        else:
            a_temp.run_multiple_episode()
        agent_dict[algorithm] = a_temp
    return agent_dict

def plot_results(agent_dict):
    plt.close()
    for algorithm, agent in agent_dict.items():
        plt.plot(range(0, agent.evaluate_every * len(agent.avg_rewards), agent.evaluate_every), agent.avg_rewards, label=algorithm)
    plt.xlabel('Training episodes')
    plt.ylabel('Average reward on {} episodes'.format(agent.episodes_to_evaluate))
    plt.title('Average reward during training for different algorithms')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    agent_dict = train_and_evaluate_agents(args)
    plot_results(agent_dict)


    
