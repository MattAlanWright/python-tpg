from numpy.random import randint
from utils import weightedCoinFlip
from copy import deepcopy

from Program import Program

def configureLearner(atomic_action_range = 18):
    Learner.ATOMIC_ACTION_RANGE = atomic_action_range

class Learner:
    '''A Learner is a light wrapper around a Program with just enough added
    functionality to facilitate evolution (ie. learning) and bidding.
    Learners must maintain some extra concepts that Programs don't need,
    described below.

    Bidding:
    Learners must be able to bid, given a current environment state, (ie. the
    input). The bid is simply the floating point value found in the Learner's
    Program's R[0] after execution.

    Actions:
    After winning a bid, a Learner must execute some
    action. This can either be atomic, ie. simply an integer indexing into the
    action space available in the environment, or non-atomic, in which case
    the action is a pointer into another Team of Learners.

    Teams:
    As inconvenient as it is from a programming point of view, Learners need
    extensive understanding and knowledge of the Teams of which they are members.
    '''

    MAX_ACTION_RANGE = 18

    def __init__(self, action = None, learner = None):
        """Initialize new Learner. This can either be done from scratch or
        as a copy of a previous Learner, maintaining that Learner's action,
        which may be a pointer to a Team.

        Be wary of using deepcopy! The temptation to copy Learners via deepcopy
        was there, but this is a mistake since it will create copies of any Team
        pointed to by self.action. On the other hand, copying Programs via
        deepcopy is correct and a conveneient way to ensure that the new Program
        gets its own copy of the list of instructions.
        """

        # Set default value. This is so that setAction() will function properly
        # when checking the current self.action type. In general, the action
        # should always be set via setAction()
        self.action = 0

        # This counter keeps track of how many Teams hold a pointer to
        # this Learner, ie. how many Teams this Learner is a member of.
        self.num_referencing_teams = 0

        if learner is None:
            # Create Program associated with Learner
            self.program = Program()

            self.setAction(action)

            if action is None:
                print("WARNING - Learner::init - No Learner and no Action")
                # Assign Learner's action value
                self.setAction(randint(0, Learner.ATOMIC_ACTION_RANGE))

        else:
            # Make a copy of the other Learner's Program
            self.program = deepcopy(learner.program)

            # Copy the other Learner's action, whether it's atomic or not
            self.setAction(learner.action)

            # If new action is a Team pointer, update that Team's number of
            # referencing Learners
            if not self.isActionAtomic():
                self.action.incrementNumReferencingLearners()


    def incrementNumReferencingTeams(self):
        self.num_referencing_teams += 1


    def decrementNumReferencingTeams(self):
        self.num_referencing_teams -= 1


    def getNumReferencingTeams(self):
        return self.num_referencing_teams


    def isActionAtomic(self):
        from Team import isTeam

        if isTeam(self.action):
            return False
        elif isinstance(self.action, int):
            return True
        else:
            print("WARNING - Learner::isActionAtomic - Action is not Team or int")
            print("          type(self.action) =", type(self.action))
            return False


    def bid(self, input):
        """Submit a bid to have this Learner's action taken."""
        self.program.execute(input)
        return self.program.registers[0]


    def act(self, input, visited):
        """Perform action. If the action is atomic (ie. an integer) then return
        the Learner's integer action. If the action is not atomic (ie. it is
        a pointer to a Team), then call that Team's act() method.
        """
        if self.isActionAtomic():
            return self.action
        else:
            return self.action.act(input, visited)


    def setAction(self, new_action):
        """Assign an action to this Learner."""

        # Perform necessary bookkeeping given the action being relinquished.
        # If the current action is a Team pointer, decrement that Team's
        # referencing Learner count.
        if not self.isActionAtomic():
            self.action.decrementNumReferencingLearners()

        # If new action is atomic, simply set the action to new_action. Otherwise
        # perform bookkeeping on the new Team being pointed to before assigning
        # it to this Learner's action.
        from Team import isTeam
        if isTeam(new_action):
            new_action.incrementNumReferencingLearners()
            self.action = new_action
        else:
            self.action = int(new_action)


    def mutateProgram(self):
        """Mutate this Learner's Program."""
        self.program.mutate()
