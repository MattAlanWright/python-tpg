from numpy.random import choice, randint
import numpy as np
import multiprocessing as mp
import pickle
import gym
import time
import cv2

from Learner import Learner
from Team import Team
from Agent import Agent
import Memory

from Learner import configureLearner

from utils import weightedCoinFlip

# Action-selection options
TRAINER_ACTION_STANDARD         = 0 # Standard TPG action-selection
TRAINER_ACTION_REAL_VALUED      = 1 # Memory -> network -> real-valued action selection

def configureTrainer(
    atomic_action_range  = 18,      # Actions will be selection from [0, atomic_action_range)
    p_del_learner        = 0.1,
    p_add_learner        = 0.1,
    p_mut_learner        = 0.1,
    p_mut_learner_action = 0.1,
    p_atomic             = 0.6,
    r_size               = 200,     # Root Team population size
    percent_keep         = 0.30,    # Percentage of Root Teams to keep each generation
    env_name             = 'CartPole-v0',
    trainer_name         = "",      # Pass name to save Trainer
    agent_name           = "",      # Pass name to save best Agent
    verbose              = False,
    env_seed             = -1,      # -1 to use random seed
    do_fast_track        = False,   # True to skip re-evaluation of Agents for some generations
    max_eval_skips       = 0,       # Number of generations to skip re-evaulation

    training_results_nparray_name = "", # Pass string to save fitness scores
    
    action_selection    = TRAINER_ACTION_STANDARD):

    Trainer.P_DEL_LEARNER        = p_del_learner
    Trainer.P_ADD_LEARNER        = p_add_learner
    Trainer.P_MUT_LEARNER        = p_mut_learner
    Trainer.P_MUT_LEARNER_ACTION = p_mut_learner_action
    Trainer.P_ATOMIC             = p_atomic
    Trainer.ATOMIC_ACTION_RANGE  = atomic_action_range
    Trainer.R_SIZE               = r_size
    Trainer.PERCENT_KEEP         = percent_keep
    Trainer.ENV_NAME             = env_name
    Trainer.TRAINER_NAME         = trainer_name
    Trainer.AGENT_NAME           = agent_name
    Trainer.VERBOSE              = verbose
    Trainer.ENV_SEED             = env_seed
    Trainer.DO_FAST_TRACK        = do_fast_track
    Trainer.MAX_EVAL_SKIPS       = max_eval_skips
    Trainer.ACTION_SELECTION     = action_selection

    Trainer.TRAINING_RESULTS_NPARRAY_NAME = training_results_nparray_name

    # Configure similar fields in other classes
    configureLearner(atomic_action_range = atomic_action_range)


class Trainer:

    ATOMIC_ACTION_RANGE = 18

    # Mutation probabilities
    P_DEL_LEARNER        = 0.1
    P_ADD_LEARNER        = 0.1
    P_MUT_LEARNER        = 0.1
    P_MUT_LEARNER_ACTION = 0.1
    P_ATOMIC             = 0.6

    # Do we need to downsample the screen?
    PLAYING_ATARI        = False

    # Environment name
    ENV_NAME             = 'CartPole-v1'
    ENV_SEED             = -1

    # Save names
    AGENT_NAME           = ""
    TRAINER_NAME         = ""

    # Evolution configuration
    R_SIZE               = 200
    PERCENT_KEEP         = 0.30

    # Fast-tracked evolution
    DO_FAST_TRACK        = False

    # Results and log file names
    VERBOSE                       = False
    TRAINING_RESULTS_NPARRAY_NAME = ""

    # Action-selection type
    ACTION_SELECTION = TRAINER_ACTION_STANDARD

    def __init__(self):
        # Team populations
        self.team_pop      = []
        self.agent_pop     = []

        # Learner population
        self.learner_pop   = []

        # Score records
        self.avg_scores = []
        self.top_scores = []


    def initializeTeam(self):
        """Create a new Team and add two new Learners with different atomic
        actions. These Learners are added to the Learner population, and the
        Team is added to both the Team and root Team populations.
        """

        # Create two new Learners with different atomic actions
        a1 = randint(0, Trainer.ATOMIC_ACTION_RANGE)
        a2 = randint(0, Trainer.ATOMIC_ACTION_RANGE)
        while a1 == a2:
            a2 = randint(0, Trainer.ATOMIC_ACTION_RANGE)

        l1 = Learner(action = a1)
        l2 = Learner(action = a2)

        # Create new Team
        team = Team()

        # Add Learners to Team
        team.addLearner(l1)
        team.addLearner(l2)

        # Add Learners to Learner population
        self.learner_pop.append(l1)
        self.learner_pop.append(l2)

        # Add Team to Team populations. Note that all new Teams are, by
        # definition, root teams
        self.team_pop.append(team)


    def fillTeamsInPopulation(self):
        """Given a full Learner and Team population, where each Team points to
        two unique Learners, randomly add existing Learners to each Team until
        that Team reaches its full complement.
        """
        for team in self.team_pop:
            while len(team.learners) < team.max_size:
                learners = [l for l in self.learner_pop if l not in team.learners]
                if len(learners) == 0:
                    break
                else:
                    learner = choice(learners)
                    team.addLearner(learner)


    def initialize(self):
        """Create the initial complement of Teams and the Learner population."""
        r_keep = int(Trainer.R_SIZE * Trainer.PERCENT_KEEP)

        for i in range(r_keep):
            self.initializeTeam()
        self.fillTeamsInPopulation()

        # Double check that things are working as expected and that we now
        # have PERCENT_KEEP * R_SIZE Teams and that these are all root Teams.
        # Also, ensure that we have 2 * PERCENT_KEEP * R_SIZE Learners.
        if len(self.team_pop) != r_keep:
            print("WARNING - Trainer::initialize - len(self.team_pop) != r_keep")
            print("                                len(self.team_pop) =", len(self.team_pop))

        if self.getNumRootTeams() != r_keep:
            print("WARNING - Trainer::initialize - getNumRootTeams() != r_keep")
            print("                                getNumRootTeams() =", self.getNumRootTeams())

        if len(self.learner_pop) != 2 * r_keep:
            print("WARNING - Trainer::initialize - len(self.learner_pop) != 2 * r_keep")
            print("                                len(self.learner_pop) =", len(self.learner_pop))


    def generation(self):
        """Refill the root Team population up to R_SIZE root Teams. This is done
        as the first step of every iteration of evolution to ensure that we are
        always evaluating R_SIZE root Teams.
        """

        # Re-count and re-collect all root Teams from the main Team
        # population into self.agent_pop
        self.updateAgentPopulation()

        # Generate new root Teams as variations of other Teams
        while self.getNumRootTeams() < Trainer.R_SIZE:

            # Randomly select parent
            parent = choice(self.agent_pop)

            # Copy the parent Team and perform mutation. Note that mutation
            # may result in the creation of new root Teams
            child = Team(parent.team)
            self.mutateTeam(child)

            # Add new Team to the Team populations
            self.team_pop.append(child)

        # Since mutation can theoretically cause new root Teams to be created,
        # run a check for this just out of curiosity.
        if self.getNumRootTeams() != Trainer.R_SIZE:
            print("NOTE - Trainer::generation - self.getNumRootTeams() != Trainer.R_SIZE")
            print("                             self.getNumRootTeams() =", self.getNumRootTeams())


    def selection(self):
        """During evolution, after evaluating all agents (ie. root teams), delete
        the PERCENT_KEEP worst-performing Teams.
        """

        # Sort root Teams from best to worst
        ranked_agents = sorted(self.agent_pop, key=lambda rt : rt.team.fitness, reverse=True)

        # Save trainer and top agent so far
        self.save()
        if self.AGENT_NAME == "":
            ranked_agents[0].save(Trainer.ENV_NAME)
        else:
            ranked_agents[0].save(Trainer.AGENT_NAME)

        # Sanity check: There should always be R_SIZE root Teams at this point
        if len(self.agent_pop) != Trainer.R_SIZE:
            print("WARNING - Trainer::selection - len(self.agent_pop) != Trainer.R_SIZE")
            print("                               len(self.agent_pop) = ", len(self.agent_pop))

        # Calculate the number of root Teams to retain
        num_keep = int(Trainer.PERCENT_KEEP * Trainer.R_SIZE)

        # Grab slice of sorted root Team references to delete
        agents_to_delete = ranked_agents[num_keep:]

        # Clean all root Teams in the Teams-to-delete list
        # Note: Still need to clean the population of Learners which
        # may now contain orphans and update the root Team population.
        for agent in agents_to_delete:

            team = agent.team

            # Safety and sanity check: These should ALL be root Teams
            if team.getNumReferencingLearners() != 0:
                print("WARNING - Trainer::selection - A non-root Team is being deleted!")

            team.removeLearners()
            self.team_pop.remove(team)

        # Clean up orphanced Learners after Team removal
        self.cleanOrphanedLearners()


    def evaluateAgent(self, agent, env, num_episodes, num_frames):
        """Evaluate an agent over some number of episodes in a given environment 'env'
        """

        # Skip agents that have already been evaluated, up to MAX_EVAL_SKIPS times
        if agent.team.fitness is not None and Trainer.DO_FAST_TRACK:
            if agent.team.num_skips < Trainer.MAX_EVAL_SKIPS:
                agent.team.num_skips += 1
                return
            else:
                agent.team.num_skips = 0

        # Track scores across episodes
        scores = []

        for ep in range(num_episodes):

            # Reset the environment for a new episode
            if Trainer.ENV_SEED >= 0:
                env.seed(Trainer.ENV_SEED)

            state = env.reset()

            # Reset the score for this episode
            score = 0

            # Loop for num_frames frames or until the episode is done
            for fr in range(num_frames):

                # Retrieve the Agent's action. Standard action selection. Even if using a non-standard
                # action-selection mechanism, this needs to be executed to operate on the Memory.
                action = agent.act(state.reshape(-1))

                if Trainer.ACTION_SELECTION != TRAINER_ACTION_STANDARD:
                    
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

            # Record this episode's score
            scores.append(score)

        # Assign final fitness to Agent
        agent.team.fitness = np.mean(scores)


    def evaluation(self, env, num_episodes, num_frames):
        # Re-count and re-collect all root Teams from the main Team population
        self.updateAgentPopulation()

        scores = []

        # Get fitness of all root Teams
        for i, agent in enumerate(self.agent_pop):

            # Evaluate the agent in the current task/environment
            self.evaluateAgent(agent, env, num_episodes, num_frames)
            scores.append(agent.team.fitness)

        # Optionally save results
        self.avg_scores.append(np.mean(scores))
        self.top_scores.append(np.max(scores))
        if Trainer.TRAINING_RESULTS_NPARRAY_NAME != "":
            np.save("{}_avg.npy".format(Trainer.TRAINING_RESULTS_NPARRAY_NAME),
                    np.array(self.avg_scores),
                    allow_pickle=True)
            np.save("{}_top.npy".format(Trainer.TRAINING_RESULTS_NPARRAY_NAME),
                    np.array(self.top_scores),
                    allow_pickle=True)

        # Print average and top score this generation
        if Trainer.VERBOSE:
            print("    Average score this generation:", int(np.mean(scores)))
            print("    Top score this generation:", int(np.max(scores)))


    def evolve(self, env, num_generations, num_episodes, num_frames):
        """Perform the TPG evolution algorithm."""
        for gen in range(num_generations):

            if Trainer.VERBOSE:
                print("Generation:", gen)

            # Generate new root Teams
            self.generation()

            # Evaluate current agents
            self.evaluation(env, num_episodes, num_frames)

            # Perform selection
            self.selection()

        # Return to top-performing agent. Typically not used, but nice to have
        ranked_agents = sorted(self.agent_pop, key=lambda rt : rt.team.fitness, reverse=True)
        return ranked_agents[0]


    def removeLearners(self, team):
        # Probabalistically delete Learners
        p = Trainer.P_DEL_LEARNER
        while weightedCoinFlip(p):
            p *= Trainer.P_DEL_LEARNER

            # Ensure the Team maintains at least 2 Learners
            if len(team.learners) <= 2:
                break

            # Choose random Learner from Team's Learners, ensuring that there is at least one atomic action
            # Learner remaining in the Team
            num_atomic_actions_in_team = team.countAtomicActions()

            # Sanity check, there should always be at least 1 atomic action
            if num_atomic_actions_in_team == 0:
                print("WARNING - Trainer::removeLearners - No atomic actions in Team!")

            learners = [l for l in team.learners if num_atomic_actions_in_team > 1 or not l.isActionAtomic()]
            learner = choice(learners)

            # Remove Learner from Team
            team.removeLearner(learner)

            # Sanity check
            if team.countAtomicActions() == 0:
                print("WARNING - Trainer::removeLearners - We removed the last atomic Learner!")


    def addLearners(self, team):
        # Probabalistically add new Learners
        p = Trainer.P_ADD_LEARNER
        while weightedCoinFlip(p):
            p *= Trainer.P_ADD_LEARNER

            # Ensure the Team maintains at least 2 Learners
            if len(team.learners) >= Team.MAX_TEAM_SIZE:
                break

            # Choose random Learner from ALL Learners to add to the Team
            learners = [l for l in self.learner_pop if l not in team.learners and l.action is not self]
            if len(learners) == 0:
                break
            learner = choice(learners)

            # Remove Learner from Team
            team.addLearner(learner)


    def mutateLearners(self, team):
        # Since we will remove Learners from the Team while iterating over them,
        # grab a copy of the list of Learner references. We will iterate through
        # the copy while mutating the original list
        learners = team.learners.copy()

        # Probabalistically mutate Learners in the Team
        for learner in learners:

            # Most Learners will be skipped and not mutated
            if not weightedCoinFlip(Trainer.P_MUT_LEARNER):
                continue

            # Sanity check
            if team.countAtomicActions() == 0:
                print("WARNING - Trainer::mutateLearners - No atomic actions in Team!")

            # If this Team only has one Learner with an atomic action and it
            # happens to be this Learner, remove the possibility of setting the
            # new action to a Team pointer
            p_atomic = Trainer.P_ATOMIC
            if learner.isActionAtomic() and team.countAtomicActions() == 1:
                p_atomic = 1.0

            # Copy Learner. The copy will be mutated and added to the pool of
            # Learners. This is done to ensure that, in the event that this Learner
            # is also used to great success in another Team, that other Team
            # still points to the original functional Learner.
            learner_prime = Learner(learner=learner)

            # Remove original Learner from Team and replace it with the new one.
            team.removeLearner(learner)
            team.addLearner(learner_prime)

            # Mutate the new Learner
            learner_prime.mutateProgram()

            # Probabalistically mutate the new Learner's action
            if weightedCoinFlip(Trainer.P_MUT_LEARNER_ACTION):
                self.mutateLearnerAction(learner_prime, p_atomic)

            # Sanity check
            if team.countAtomicActions() == 0:
                print("WARNING - Trainer::mutateLearners - No atomic actions in Team after mutation!")

            # Add new Learner to the pool of Learners
            self.learner_pop.append(learner_prime)


    def mutateLearnerAction(self, learner, p_atomic):
        if weightedCoinFlip(p_atomic):
            learner.setAction(randint(0, Trainer.ATOMIC_ACTION_RANGE))
        else:
            team = choice(self.team_pop)
            learner.setAction(team)


    def mutateTeam(self, team):
        """Perform variation operators on the Team level."""
        self.removeLearners(team)
        self.addLearners(team)
        self.mutateLearners(team)


    def cleanOrphanedLearners(self):
        """Delete all Learners that are no longer owned by any Team."""

        # Before deleting Learners, ensure that if any Learners that are about to be
        # deleted point to a Team as their action, then that Team's count of
        # referincing Learners is decremented.
        for learner in self.learner_pop:
            if learner.getNumReferencingTeams() == 0 and not learner.isActionAtomic():
                learner.action.decrementNumReferencingLearners()

        # Remove all orphaned Learners from the Learner population
        self.learner_pop = [l for l in self.learner_pop if not l.getNumReferencingTeams() == 0]


    def getNumRootTeams(self):
        num_root_teams = 0
        for team in self.team_pop:
            if team.getNumReferencingLearners() == 0:
                num_root_teams += 1
        return num_root_teams


    def updateAgentPopulation(self):
        self.agent_pop = [Agent(t) for t in self.team_pop if t.getNumReferencingLearners() == 0]

    def save(self):
        if Trainer.TRAINER_NAME != "":
            pickle.dump(self, open(Trainer.TRAINER_NAME + ".trainer", 'wb'))

def loadTrainer(fname):
    trainer = pickle.load(open(fname, 'rb'))
    trainer.updateAgentPopulation()
    max_team_id = 0
    for agent in trainer.agent_pop:
        if agent.id > max_team_id:
            max_team_id = agent.id
    Team.ID_COUNT = max_team_id + 1

    return trainer
