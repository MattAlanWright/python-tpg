import gym
import pickle
import sys
import numpy as np
import time
import cv2
import argparse
sys.path.insert(0, '../')

from Agent import loadAgent

def run(arguments):
    if len(arguments) == 0:
        print("ERROR - No arguments given to main")
        sys.exit(0)

    parser = argparse.ArgumentParser(description='Load a TPG agent into an environment and render the results.')
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--agent', dest='agent_fname', type=str, help='Previously saved Agent', default="")
    parser.add_argument('--seed', dest='seed', type=int, help='Seed for environment', default=-1)
    parser.add_argument('--action_type', dest='action_type', type=int, help='0 = standard, 1 = Memory + discrete, 2 = Memory, real-valued', default=0)
    args = parser.parse_args(arguments)

    agent = loadAgent(args.agent_fname)

    env = gym.make(args.env)

    if args.seed > -1:
        env.seed(args.seed)

    state = env.reset()

    score = 0

    for i in range(18000):
        env.render()

        # Retrieve the Agent's action
        action = agent.act(state.reshape(-1))

        if args.action_type != 0: # If NOT standard TPG actions
            
            # Act on memory
            state = Memory.get().clip(-20.0, 20.0).sum() * 0.1
            
            # Continuous actions:
            mu = np.tanh(state)
            action = np.random.normal(mu, 0.1)

        # Perform action and get next state
        state, reward, is_done, debug = env.step(action)

        score += reward

        if is_done:
            break

        time.sleep(0.01)

    print("Final score: {}".format(score))

    env.close()

if __name__ == "__main__":
    run(sys.argv[1:])