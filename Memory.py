# Interface to a global Memory structure. This structure is a matrix
# that Learners can index into during read or write operations. By
# default, the structure is a 5 x 8 matrix of 32-bit floats but these
# can be changed via the following:
#
# set_size(new_size)
# set_type(new_type)
# configure()
#
# The final call to configure is necessary for the changes to take effect.
# During use, only the 'get()' function should be called. This returns a
# matrix that can be operated on normally. The main job of this file is to
# make global access to that matrix convenient.

import numpy

mem_size = (5, 8)
mem_type = numpy.float32
memory = numpy.zeros(mem_size, dtype=mem_type)

def set_size(new_size):
    global mem_size
    mem_size = new_size

def get_size():
    global mem_size
    return mem_size

def get_length():
    global mem_size
    return mem_size[0] * mem_size[1]

def set_type(new_type):
    global mem_type
    mem_type = new_type

def configure():
    global mem_size
    global memory

    memory = numpy.zeros(mem_size, dtype=mem_type)

def get():
    global memory
    return memory