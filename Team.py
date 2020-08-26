import numpy as np
from numpy.random import randint

from Learner import Learner

# Used to check if an action is atomic or a Team reference
def isTeam(t):
    return isinstance(t, Team)

class Team:

    # Team configuration
    MAX_TEAM_SIZE = 8

    ID_COUNT = 0

    def __init__(self, team = None):

        # Assign unique ID to each new Team
        self.id = Team.ID_COUNT
        Team.ID_COUNT += 1

        # Number of generations that fitness re-evaluation is skipped
        self.num_skips = 0

        # Initialize this Team's Learner complement to an empty list
        self.learners = []

        # Track the number of Learners whose action is a pointer to this Team
        self.num_referencing_learners = 0

        # Re-calculated every time the Team is a root team and acts in an environment
        self.fitness = None

        # List of this Team's Learners' register 0 values
        self.registers = None

        # If the new Team is being created as a copy of another Team, copy
        # the list of Learners (specifically, references to Learners)
        if team is not None:
            # Copy max_size attribute
            self.max_size = team.max_size

            # Copy Learner references
            for learner in team.learners:
                self.addLearner(learner)
        else:
            # Determine the maximum size (ie. number of Learners) of this Team
            # Each Team must include at least two Learners
            self.max_size = randint(2, Team.MAX_TEAM_SIZE + 1)


    def act(self, state, visited=set()):
        """Return the atomic action suggested by this team given some state.
        If the Team's action is a pointer to another Team, we continue to
        traverse the graph and propogate the final atomic action back up.
        """

        # Reset registers before acting
        self.registers = np.zeros(len(self.learners), np.float32)

        # Sanity and safety check. Should be removed for better performance
        if self.countAtomicActions() == 0:
            print("WARNING - Team::act - No atomic actions in Team!")

        # Track all Teams that are visited during acting. This is done to
        # prevent cycles while traversing the graph. Adding Teams to a Python
        # set and checking for their inclusion are both constant time.
        visited.add(self)

        # Find the Learner with the highest bid
        top_learner = None
        top_bid = 0

        for i, learner in enumerate(self.learners):

            # Skip Learners that point to an already-visited Team
            if not learner.isActionAtomic() and learner.action in visited:
                continue

            # Let the current Learner submit a bid to act
            bid = learner.bid(state)
            self.registers[i] = bid
            if bid > top_bid or top_learner is None:
                top_bid = bid
                top_learner = learner

        # Note: The following is a call to Learner::act not another call to
        # Team::act (ie. what we're in right now). Adding this reminder for
        # future readers and myself, since this is easy to lose track of. This
        # will either return an atomic action or call into a pointed-to-Team's
        # act function.
        return top_learner.act(state, visited)


    # Add Learner to Team's complement and perform book-keeping
    def addLearner(self, learner):
        """Append a new Learner to the Team."""

        # Confirm learner is NOT a member of the Team. This is here for sanity
        # and bug-finding only, this check should never be True!!
        if learner in self.learners:
            print("WARNING - Team::addLearner - Learner already a member of Team")
            return

        # Update the Learner's count of referencing teams
        learner.incrementNumReferencingTeams()

        # Add the Learner to the Team's complement
        self.learners.append(learner)


    # Remove Learner from Team's complement and perform book-keeping
    def removeLearner(self, learner):
        """Remove Learner from the Team."""

        # Confirm learner is a member of the Team. This is here for sanity
        # and bug-finding only, this check should never be True!!
        if learner not in self.learners:
            print("WARNING - Team::removeLearner - Learner not a member of Team")
            return

        # Update the Learner's count of referencing teams
        learner.decrementNumReferencingTeams()

        # Remove the Learner from the Team
        self.learners.remove(learner)


    def removeLearners(self):
        """Remove all Learners from the Team. This includes updating the Learners'
        counts of number of referencing Teams. After this method is called, the
        Team's 'learners' member is reset to an empty list. This is only done for safety and
        sanity; in practice this method is only called when a Team is being deleted
        so it is unikely the list will be used in the future.
        """
        for learner in self.learners:
            # Update the Learner's count of refering teams
            learner.decrementNumReferencingTeams()

        self.learners = []


    def incrementNumReferencingLearners(self):
        self.num_referencing_learners += 1


    def decrementNumReferencingLearners(self):
        self.num_referencing_learners -= 1


    def getNumReferencingLearners(self):
        return self.num_referencing_learners


    def countAtomicActions(self):
        num_atomic_actions = 0
        for learner in self.learners:
            if learner.isActionAtomic():
                num_atomic_actions += 1
        return num_atomic_actions
