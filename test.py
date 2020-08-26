# Test script. This script loads a saved Agent, runs it through some number of
# episodes in a selected environment and records various data as the Agent acts.
# This program is typically run as a command line program. For example
#
# $ python test.py --env CartPole-v1 --agent cp.agent
#
# Will test the Agent 'cp.agent' in the CartPole-v1 environment.

import gym
import pickle
import sys
import numpy as np
import time
import argparse

from pickle import dump
from sklearn.preprocessing import StandardScaler

import Memory
from Agent import loadAgent

def run(arguments):
    if len(arguments) == 0:
        print("ERROR - No arguments given to main")
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Perform TPG evolution for a given environment.')
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--agent', dest='agent_fname', type=str, help='Previously saved Agent', default="")
    parser.add_argument('--eps', dest='eps', type=int, help='Number of random episodes', default=500)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed for environment', default=-1)
    parser.add_argument('--test_results_savename', dest='test_results_savename', type=str, help='Name under which to save the test results', default="")
    parser.add_argument('--verbose', dest='verbose', type=bool, help="Print results to the console as we evolve", default=False)
    parser.add_argument('--action_type', dest='action_type', type=int, help='0 = Standard, 1 = Real-valued', default=0)
    args = parser.parse_args(arguments)

    agent = loadAgent(args.agent_fname)

    env = gym.make(args.env)

    scores = []

    if args.seed > -1:
        env.seed(args.seed)

    states = []

    for i in range(args.eps):
        state = env.reset()
        score = 0
        for j in range(18000):

            # Retrieve the Agent's action
            action = agent.act(state.reshape(-1))

            if args.action_type != 0:
                
                # Act on memory
                state = Memory.get().clip(-20.0, 20.0).sum() * 0.1
            
                # Continuous actions:
                mu = np.tanh(state)
                action = np.random.normal(mu, 0.1)

            # Perform action and get next state
            state, reward, is_done, debug = env.step(action)

            # Keep running tally of score
            score += reward

            if is_done:
                break

        scores.append(score)

    env.close()

    # Optionally save scores
    scores = np.array(scores, dtype=np.float32)
    if args.test_results_savename != "":
        np.save("{}.npy".format(args.test_results_savename),
                scores,
                allow_pickle=True)

    # Optionally print results
    if args.verbose:
        print("Average score over {} episodes: {}".format(args.eps, scores.mean()))

    return scores

if __name__ == "__main__":
    run(sys.argv[1:])