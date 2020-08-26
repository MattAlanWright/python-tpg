import numpy as np
from scipy.special import softmax

import pickle

from Team import Team

class Agent:

    def __init__(self, team):
        self.team       = team
        self.id         = team.id


    # Take an action. Called every timestep, passed the current environment state
    def act(self, state):
        self.reset()
        return self.team.act(state, visited=set())

    # Save a complete Agent structure
    def save(self, name="agent"):
        pickle.dump(self, open(name + ".agent", 'wb'))

    def reset(self):
        self._reset([self.team], visited_teams=set())

    # Reset all Teams' 'registers' structures to 0. Reminder that for a Team,
    # the 'registers' structure is a copy of its learners' register 0 values.
    def _reset(self, teams_to_visit, visited_teams=set()):

        # Record new teams that need to be visited
        new_teams_to_visit = []

        # Run through all teams at this level of the graph
        for team in teams_to_visit:

            # Skip visited teams
            if team in visited_teams:
                continue

            # Add to visited set
            visited_teams.add(team)

            # Reset team's registers
            team.registers = np.zeros(len(team.learners), np.float32)

            # Get all learners' registers and team pointers
            for learner in team.learners:
                if not learner.isActionAtomic() and learner.action not in new_teams_to_visit:
                    new_teams_to_visit.append(learner.action)

        if len(new_teams_to_visit) == 0:
            return
        else:
            return self._reset(new_teams_to_visit, visited_teams)


    # Return a single vector containing all Teams' 'registers' concatenated
    def getRegisters(self, do_softmax=False):
        return self._getRegisters([self.team], do_softmax=do_softmax, visited_teams=set(), registers=np.array([], dtype=np.float32))


    def _getRegisters(self, teams_to_visit, do_softmax=False, visited_teams=set(), registers=np.array([], dtype=np.float32)):

        # Record new teams that need to be visited
        new_teams_to_visit = []

        # Run through all teams at this level of the graph
        for team in teams_to_visit:

            # Skip visited teams
            if team in visited_teams:
                continue

            # Add to visited set
            visited_teams.add(team)

            # Optionally softmax the team's registers
            if do_softmax:
                current_regs = softmax(team.registers.copy())
            else:
                current_regs = team.registers.copy()

            # Get team's results
            registers = np.concatenate((registers, current_regs))

            # Get all learners' registers and team pointers
            for learner in team.learners:
                if not learner.isActionAtomic() and learner.action not in new_teams_to_visit:
                    new_teams_to_visit.append(learner.action)

        if len(new_teams_to_visit) == 0:
            return registers.reshape(-1)
        else:
            return self._getRegisters(new_teams_to_visit, do_softmax, visited_teams, registers)


# Load a saved Agent structure
def loadAgent(fname):
    agent = pickle.load(open(fname, 'rb'))
    return agent
