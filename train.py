# Agent training script. This script lets you run the TPG agent training routine and
# optionally save lots of different intermediate and final results. It's also an example
# of the steps you need to take if you want to train an Agent on your own outside of
# this script. In general this script is run from the command line. For example:
#
# $ python train.py --env CartPole-v1 --agent_savename cp
#
# will train a TPG Agent to play CartPole-v1 and save the resulting Agent as 'cp.agent'.

import gym
import numpy as np
from functools import reduce

from Trainer import Trainer, configureTrainer, loadTrainer
from Program import configureProgram

import os
import sys
import argparse

def run(arguments):
    if len(arguments) == 0:
        print("ERROR - No arguments given to main")
        sys.exit(0)

    # Setup the command line parsing to read the environment title
    parser = argparse.ArgumentParser(description='Perform TPG evolution for a given environment.')
    parser.add_argument('--env', dest='env', type=str, help='OpenAI environment', default="CartPole-v1")
    parser.add_argument('--trainer', dest='trainer_fname', type=str, help='Previously saved Trainer object', default="")
    parser.add_argument('--trainer_savename', dest='trainer_savename', type=str, help='Name under which to save the Trainer object', default="")
    parser.add_argument('--agent_savename', dest='agent_savename', type=str, help='Name under which to save the top Agent object', default="")
    parser.add_argument('--training_results_savename', dest='training_results_savename', type=str, help='Name under which to save the training results', default="")
    parser.add_argument('--generations', dest='num_generations', type=int, help='Number of generations', default=500)
    parser.add_argument('--episodes', dest='num_episodes', type=int, help='Number of episodes per agent at each generation', default=1)
    parser.add_argument('--pop', dest='r_size', type=int, help='Number of agents (root teams) per generation', default=200)
    parser.add_argument('--frames', dest='num_frames', type=int, help='Max number of frames per episode', default=18000)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed for environment', default=-1)
    parser.add_argument('--verbose', dest='verbose', type=bool, help="Print results to the console as we evolve", default=False)
    parser.add_argument('--fast', dest='fast', type=bool, help="Set to True to skip re-evaluating agents", default=False)
    parser.add_argument('--skips', dest='skips', type=int, help='Maximum number of times an agent can skip re-evaluation', default=0)
    parser.add_argument('--action_type', dest='action_type', type=int, help='0 = Standard, 1 = Real-valued', default=0)
    args = parser.parse_args(arguments)

    # Get environment details
    env = gym.make(args.env)

    if args.action_type == 0:
        num_actions = env.action_space.n
    elif args.action_type == 1:
        num_actions = 100   # This number is meaningless when choosing real-valued actions
                            # but the Team initialization doesn't like anything less than 2.
                            # In fact, the bigger the number, the faster Team intialization.
    else:
        print("Invalid action_type argument {}".format(args.action_type))

    input_size  = reduce(lambda x, y: x * y, env.observation_space.shape)

    configureTrainer(
        atomic_action_range  = num_actions,
        p_del_learner        = 0.7,
        p_add_learner        = 0.7,
        p_mut_learner        = 0.3,
        p_mut_learner_action = 0.6,
        p_atomic             = 0.5,
        r_size               = args.r_size,
        percent_keep         = 0.40,
        env_name             = args.env,
        trainer_name         = args.trainer_savename,
        agent_name           = args.agent_savename,
        verbose              = args.verbose,
        env_seed             = args.seed,
        do_fast_track        = args.fast,
        max_eval_skips       = args.skips,
        action_selection     = args.action_type,
        training_results_nparray_name = args.training_results_savename)

    configureProgram(
        p_add         = 0.6,
        p_del         = 0.6,
        p_mut         = 0.6,
        p_swap        = 0.6,
        max_prog_size = 128,
        min_prog_size = 8,
        num_registers = 8,
        input_size    = input_size)

    # If trainer filename was passed in, re-create Trainer. Otherwise, create
    # a new one from scratch
    if args.trainer_fname == "":
        # Create Trainer
        trainer = Trainer()

        # Perform TPG initialization (ie. create initial Team and Learner populations)
        trainer.initialize()
    else:
        # Load Trainer
        trainer = loadTrainer(args.trainer_fname)

    # Try to generate an agent
    trainer.evolve(env,
                num_generations=args.num_generations,
                num_episodes=args.num_episodes,
                num_frames=args.num_frames)

if __name__ == "__main__":
    run(sys.argv[1:])