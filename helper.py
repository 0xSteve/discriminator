'''Some helper files for this work.'''

from random import uniform as rand
import numpy as np
from numpy import linalg as LA
from math import *
import matplotlib.pyplot as plt


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

# THIS IDEA DID NOT WORK AT ALL! Somehow doing them together did?

# def make_Z(size=200, dimensions=3):
#     '''Create an array of Normally distributed points.'''
#     Z = normal_point(dimensions)  # Will become a non-empty numpy array.

#     for i in range(size - 1):
#         point = normal_point(dimensions)
#         # add the new point on axis 1 to create a new column.
#         Z = np.append(Z, point, axis=1)

#     return Z


# def make_X(Z, lambda_x, mean, eigvec, size=200, dimensions=3):
#     '''Given a standard normal vector Z, a mean, and eigen values and vectors of
#        variance, generate a translated normal vector X.'''

#     x = eigvec @ np.power(lambda_x, 0.5) @ Z[0, :] + mean

#     for i in range(1, size):
#         point = eigvec @ np.power(lambda_x, 0.5) @ Z[i] + mean
#         x = np.append(x, point, axis=1)

#     return x

def make_ZX(lambda_x, mean, eigvec, size=200, dimensions=3):
    '''Given a standard normal vector Z, a mean, and eigen values and vectors of
       variance, generate a translated normal vector X.'''
    z = normal_point(dimensions)

    x = eigvec @ np.power(lambda_x, 0.5) @ z + mean

    for i in range(1, size):
        pt = normal_point(dimensions)
        z = np.append(z, pt, axis=1)
        pt =  eigvec @ np.power(lambda_x, 0.5) @ pt + mean
        x = np.append(x, pt, axis=1)
    # ROFL Don't indent the return!!!!
    return x, z



def inv_sqrt(A):
    '''Perform the element wise inverse square root. Assumes m x n matrix.'''
    for i in range(len(A)):
        for j in range(len(A[1])):
            A[i][j] = math.pow(A[i][j], -0.5)

    return A


def two_class_discriminant(trd, sigma1, sigma2, mean1, mean2, p1=0.5, p2=0.5):
    '''Compute the two class discriminant for a given set of training data.'''
    typer = np.zeros(1)
    # If the training data and others are not np.array type make them np.array
    # type.
    if(type(trd) != type(typer)):
        trd = np.array(trd)

    if(type(sigma1) != type(typer)):
        trd = np.array(sigma1)

    if(type(sigma2) != type(typer)):
        trd = np.array(sigma2)

    if(type(mean1) != type(typer)):
        trd = np.array(mean1)

    if(type(mean2) != type(typer)):
        trd = np.array(mean2)

    # From our course notes
    # ax^2 +bx + c
    # xT a x => equivalent to x^2
    a = (np.linalg.inv(sigma2) - np.linalg.inv(sigma1))

    b = mean1.transpose() @ (np.linalg.inv(sigma1) - mean2.transpose()) @ np.linalg.inv(sigma2)
    # Don't specify base for math.log base e, (ln), np.log is base e
    c = log(p1 / p2) + np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))

    return (trd.transpose() @ a @ trd + b @ trd + c)

# Now to correct some problems with my diagonalization from last assignment. I
# figured it would be better to include it in my helper functions instead of
# making a lengthy setup in my main assignment. I think I may end up doing the
# same thing for plotting.
def two_class_diag(X1, M1, S1, X2, M2, S2):
    '''Solve the simultaneous diagonalization problem.'''
    # This part has the corrections to my assigment2 mistake. Mostly it was
    # an issue with dimensions in numpy. I'm used to dimensioning my vectors
    # MATLAB style, and I didn't want to be looping over and over.
    w1, v1 = np.linalg.eig(S1)
    w2, v2 = np.linalg.eig(S2)

    # make the intermediary Y...
    Y1 = v1.transpose() @ X1
    Y2 = v2.transpose() @ X2
    # Mean of Y
    My1 = v1.transpose() @ M1
    My2 = v2.transpose() @ M2
    # Mean of Z
    Mz1 = v1.transpose() @ My1
    Mz1 = v2.transpose() @ My2
    # Mean of V
    Mv1 = v1.transpose() @ Mz1
    Mv2 = v2.transpose() @ Mz2

    # Make Z
    # instead of using P1 and so forth just use the w1 and w2
    Z1 = np.diag(np.power(w1, -0.5)) @ v1.transpose() @ X1
    Z2 = np.diag(np.power(w2, -0.5)) @ v1.transpose() @ X2
    # Make Sz1, Sz2
    Sz1 = np.diag(np.power(w1, -0.5)) @ np.diag(np.power(w1, -0.5)) @ v1.transpose()
    Sz2 = np.diag(np.power(w1, -0.5)) @ v1.transpose() @ sigma2 @ v1 @ np.diag(np.power(w1, -0.5))
    # Now get the P overall
    Poa = 0 # stands for P OverAll

    # make V1 V2
    wz1, vz1 = np.linalg.eig(Sz1)
    wz2, vz2 = np.linalg.eig(Sz2)
    Sv1 = vz2.transpose() @ Sz1 @ vz2
    Sv2 = vz2.transpose() @ Sz2 @ vz2
    Poa = vz2.transpose() @ np.diag(np.power(w1, -0.5)) @ v1.transpose()
    V1 = Poa @ X1
    V2 = Poa @ X2
    # maybe return everything, or just V
    return V1, Mv1, Sv1, V2, Mv2, 

def nice_plot(omega1, omega2, dim1, dim2, label1, label2):
    title = 'Plot in the ' + label1 + ' --' + label2 + ' domains'
    dim1 -= 1  # Adjust for vector index.
    dim2 -= 1  # Adjust for vector index.
    plt.plot(omega1[dim1, :], omega1[dim2, :], 'y.', label="Class One")
    plt.plot(omega2[dim1, :], omega2[dim2, :], 'g.', label="Class Two")
    plt.title(title)
    plt.legend(loc=1)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.show()
