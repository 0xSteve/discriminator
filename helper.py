'''Some helper files for this work.'''

from random import uniform as rand
import numpy as np
from numpy import linalg as LA
import math


def normal_point(dimensions=3):
    '''Generate a Normally distributed point from a multivariate Normal
       distribution of a specified dimension, by default 3.'''
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


def make_Z(size=200, dimensions=3):
    '''Create an array of Normally distributed points.'''
    Z = normal_point(dimensions)  # Will become a non-empty numpy array.

    for i in range(size - 1):
        point = normal_point(dimensions)
        # add the new point on axis 1 to create a new column.
        Z = np.append(Z, point, axis=1)

    return Z


def make_X(Z, lambda_x, mean, eigvec, size=200, dimensions=3):
    '''Given a standard normal vector Z, a mean, and eigen values and vectors of
       variance, generate a translated normal vector X.'''
    x = eigvec @ np.power(lambda_x, 0.5) @ Z[0] @ mean

    for i in range(size - 1):
    point = eigvec @ np.power(lambda_x, 0.5) @ Z[0] @ mean


def inv_sqrt(A):
    '''Perform the element wise inverse square root. Assumes m x n matrix.'''
    for i in range(len(A)):
        for j in range(len(A[1])):
            A[i][j] = math.pow(A[i][j], -0.5)

    return A
