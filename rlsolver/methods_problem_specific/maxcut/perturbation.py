# perturbation.py
import random
import math
def choose_perturbation(omega, T, P0, Q):


    if omega >= T:
        return 'random'


    P = max(P0, math.exp(-omega / T))
    r = random.random()
    if r < P * Q:
        return 'direct1'
    if r < P:
        return 'direct2'
    return 'random'
