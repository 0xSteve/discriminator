'''Some helper files for this work.'''

from random import uniform as rand
import numpy as np
from numpy import linalg as LA
import math


def normal_vector(size):
    '''Generate a Normally distributed vector from a uniform pseudo-random
       number generator.'''
    vec = []
    for i in range(size):
        temp = 0
        a = 0
        for k in range(12):
            a = rand(0, 1)
            temp += a
        temp = temp - 6
        vec.append(temp)
    return vec


def normal_point(dimensions=3):
    '''Generate a point from a standard normal distribution.'''
    x = []
    for i in range(dimensions):
        z = 0
        for j in range(12):
            a = rand(0, 1)
            z += a
        z -= 6
        x.append([z])
    x = np.array(x)
    return x


def multi_normal_vector(size, dimensions=2):
    '''Generate a Normally distributed vector from a uniform pseudo-random
       number generator.'''

    vec = []
    for i in range(size):
        x = []
        for j in range(dimensions):
            temp = 0
            a = 0
            for k in range(12):
                a = rand(0, 1)
                temp += a
            temp = temp - 6
            x.append(temp)
        vec.append(x)
    return vec


def inv_sqrt(A):
    '''Perform the element wise inverse square root. Assumes m x n matrix.s'''
    for i in range(len(A)):
        for j in range(len(A[1])):
            A[i][j] = math.pow(A[i][j], -0.5)

    return A
