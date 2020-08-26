from numpy.random import uniform

def weightedCoinFlip(probability):
    return uniform(0.0, 1.0) < probability
