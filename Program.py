'''
Implementation of an eight-register linear genetic program to be
used in a full TPG algorithm.
'''
import Memory
import numpy as np
from numpy.random import randint
from numba import njit
import math
from utils import weightedCoinFlip

# Configure the Program class
def configureProgram(
    p_add               = 0.2,
    p_del               = 0.2,
    p_mut               = 0.2,
    p_swap              = 0.2,
    max_prog_size       = 96,
    min_prog_size       = 8,
    num_registers       = 8,
    input_size          = 0,
    max_source_index    = Memory.get_length()):

    if input_size == 0:
        print("configureProgram - WARNING - Input size set to 0")

    Program.P_ADD               = p_add
    Program.P_DEL               = p_del
    Program.P_MUT               = p_mut
    Program.P_SWAP              = p_swap
    Program.MAX_PROG_SIZE       = max_prog_size
    Program.MIN_PROG_SIZE       = min_prog_size
    Program.NUM_REGISTERS       = num_registers
    Program.INPUT_SIZE          = input_size
    Program.MAX_SOURCE_INDEX    = max_source_index


# Instruction configuration
# -------------------------
# Accessor modes
REGISTER_MODE               = 0
INPUT_MODE                  = 1
NUM_MODES                   = 2

# Operations
ADDITION_OP                 = 0
SUBTRACTION_OP              = 1
MULTIPLICATION_OP           = 2
DIVISION_OP                 = 3
COS_OP                      = 4
CONDITIONAL_OP              = 5
READ_MEM                    = 6
WRITE_MEM                   = 7
NUM_OP_CODES                = 8

MIN_RESULT                  = np.finfo(np.float32).min
MAX_RESULT                  = np.finfo(np.float32).max
DEFAULT_OP_RESULT           = np.float32(0.0)

# Indices into an instruction, where an instruction
# is a list of four integers
MODE_INDEX                  = 0
TARGET_INDEX                = 1
OP_CODE_INDEX               = 2
SOURCE_INDEX                = 3

NUM_INSTRUCTION_COMPONENTS  = 4

@njit
def clamp(num, min_val, max_val):
    num = np.float32(num)
    if np.isnan(num):
        num = np.float32(0.0)
    elif np.isinf(num) and num < 0.0:
        num = np.float32(min_val)
    elif np.isinf(num):
        num = np.float32(max_val)
    return min(max(num, min_val), max_val)


@njit
def executeInstruction(instruction, registers, source_mod_value, input, memory):
    '''Execute an individual instruction on an input array. This exists outside
    of the Program class to avoid problems with njit. This is called by the
    Program class.
    
    instruction - The current instruction being executed
    registers - Reference to the Program's register vector
    source_mod_value - List of source sizes (i.e. register length, state length)
    input - Reference to the current state or input being operated on
    memory - Reference to the global memory structure'''

    # Extract instruction components
    mode, target_index, op_code, source_index = instruction

    # Perform modulo in case we are indexing into the smaller range
    # (eg. given a source index of >8 when indexing into one of the
    # eight available registers).
    if op_code != READ_MEM and op_code != WRITE_MEM:
        source_index %= source_mod_value[mode]

    # Switch on mode (register mode vs input mode) and retrieve
    # the source value
    source = 0.0
    if op_code != READ_MEM and op_code != WRITE_MEM:
        if mode == REGISTER_MODE:
            source = registers[source_index]
        else:
            source = input[source_index]

    # Switch on operation and execute
    result = 0
    mem_write_op = False
    if op_code == ADDITION_OP:
        result = registers[target_index] + source

    elif op_code == SUBTRACTION_OP:
        result = registers[target_index] - source

    elif op_code == MULTIPLICATION_OP:
        result = source * 2.0

    elif op_code == DIVISION_OP:
        result = source / 2.0

    elif op_code == COS_OP:
        result = math.cos(source)

    elif op_code == CONDITIONAL_OP:
        if registers[target_index] < source:
            result = -registers[target_index]

    elif op_code == READ_MEM:
        source_index = source_index % (8 * 5)
        row = source_index // 8
        col = source_index % 5
        result = memory[row, col]

    elif op_code == WRITE_MEM:
        mem_write_op = True

        for c in range(5):
            # Monotonically decreasing probability
            prob = 0.7 - ((0.2 * float(c))**2.0)
            for k in range(8):
                rnd = np.random.uniform(0.0, 1.0)
                if rnd < prob:
                    memory[c, k] = registers[k]

    # Sanitize the result (clamp to save values) and assign to the target.
    # Special check because this is not done for a WRITE op
    if not mem_write_op:
        registers[target_index] = clamp(result, MIN_RESULT, MAX_RESULT)

    # Ensure result is valid floating point number. This error should
    # never occur, but better safe than sorry (the 'clamp()' function
    # above should be taking care of this. I have never ever gotten these
    # error messages and these lines should probably be deleted for better
    # performance).
    if np.isnan(registers[target_index]):
        print("WARNING - Program::executeInstruction - Encountered NaN using op", op_code)
        registers[target_index] = DEFAULT_OP_RESULT

    if np.isinf(registers[target_index]):
        print("WARNING - Program::executeInstruction - Encountered Inf using op", op_code)
        registers[target_index] = DEFAULT_OP_RESULT

    if registers[target_index] > MAX_RESULT:
        print("WARNING - Program::executeInstruction - registers[target_index] > MAX_RESULT", op_code)
        registers[target_index] = DEFAULT_OP_RESULT

    if registers[target_index] < MIN_RESULT:
        print("WARNING - Program::executeInstruction - registers[target_index] < MIN_RESULT", op_code)
        registers[target_index] = DEFAULT_OP_RESULT


class Program:

    # Mutation probabilities, common to all programs
    P_ADD   = 0.2
    P_DEL   = 0.2
    P_MUT   = 0.2
    P_SWAP  = 0.2

    # Program configuration, common to all program instances
    MAX_PROG_SIZE       = 128
    MIN_PROG_SIZE       = 8
    NUM_REGISTERS       = 8
    INPUT_SIZE          = 4
    MAX_SOURCE_INDEX    = Memory.get_length() # Often incorrect, be sure to configureProgram

    def __init__(self):
        # Pre-calculate mod value depending on source access mode
        self.source_mod_value = np.zeros(2, dtype=np.int32)
        self.source_mod_value[INPUT_MODE]    = Program.INPUT_SIZE
        self.source_mod_value[REGISTER_MODE] = Program.NUM_REGISTERS

        # Create registers
        self.registers = np.zeros(Program.NUM_REGISTERS, dtype=np.float32)

        # Initialize random instructions
        self.instructions = []
        num_new_instructions = randint(Program.MIN_PROG_SIZE, Program.MAX_PROG_SIZE + 1)
        for i in range(num_new_instructions):
            self.instructions.append(self.createRandomInstruction())

        self.instructions = np.array(self.instructions, dtype=np.int32)


    def createRandomInstruction(self):
        mode          = randint(0, NUM_MODES)
        target_index  = randint(0, Program.NUM_REGISTERS)
        op_code       = randint(0, NUM_OP_CODES)
        source_index  = randint(0, Program.MAX_SOURCE_INDEX)

        return np.array([mode, target_index, op_code, source_index])


    def execute(self, input):
        '''Execute the program's instructions on an input array.'''
        
        for instruction in self.instructions:
            self.executeInstruction(instruction, input)


    def executeInstruction(self, instruction, input):
        # Call the global 'executeInstruction' to enable numba support
        res = executeInstruction(instruction, self.registers, self.source_mod_value, input, Memory.get())
        if res != None:
            print(res)
            print("\n\n")


    def deleteRandomInstruction(self):
        index = randint(0, len(self.instructions))
        self.instructions = np.delete(self.instructions, index, axis=0)


    def addRandomInstruction(self):
        index = randint(0, len(self.instructions))
        self.instructions = np.insert(self.instructions,
                                      index,
                                      self.createRandomInstruction(),
                                      axis=0)


    def mutateRandomInstruction(self):
        instruction_index = randint(0, len(self.instructions))
        component_index   = randint(0, NUM_INSTRUCTION_COMPONENTS)

        # Figure out which component is being modified and set the upper
        # bound on the RNG.
        upper_bound = 0
        if component_index == MODE_INDEX:
            upper_bound = NUM_MODES
        elif component_index == TARGET_INDEX:
            upper_bound = Program.NUM_REGISTERS
        elif component_index == OP_CODE_INDEX:
            upper_bound = NUM_OP_CODES
        else:
            upper_bound = Program.MAX_SOURCE_INDEX

        # Mutate component
        new_val = randint(0, upper_bound)

        # While loop ensures the new component value is different from the old one
        while new_val == self.instructions[instruction_index][component_index]:
            new_val = randint(0, upper_bound)
        self.instructions[instruction_index][component_index] = new_val


    def mutate(self):
        # Random instruction deletion
        if weightedCoinFlip(Program.P_DEL) and len(self.instructions) > Program.MIN_PROG_SIZE:
            self.deleteRandomInstruction()

        # Random instruction addition (creation)
        if weightedCoinFlip(Program.P_ADD) and len(self.instructions) < Program.MAX_PROG_SIZE:
            self.addRandomInstruction()

        # Random instructions mutation
        if weightedCoinFlip(Program.P_MUT):
            self.mutateRandomInstruction()
