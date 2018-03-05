'''This is the main script for the assignment.'''

import numpy as np
# from numpy import linalg as la
from helper import *

# I made some mistakes with my assignment 2, so I've updated my
# diagonalizing code from some guaranteed correct code from Omar.
# The parts that are from assignment 2 have been corrected thanks to Omar's
# help.

# np.array(multi_normal_vector(sizeof, 3))
# Okay, this time get M1 and M2 as column vectors...
M1 = np.array([[3], [1], [4]])
M2 = np.array([[-3], [1], [-4]])
# print(np.shape(m1))
alpha = 0.1
beta = 0.2
a = 2
b = 3
c = 4

# from previous assignment.
S1 = [[a * a, beta * a * b, alpha * a * c],
          [beta * a * b, b * b, beta * b * c],
          [alpha * a * c, beta * b * c, c * c]]
S1 = np.array(np.round(S1, 2))
S2 = [[c * c, alpha * b * c, beta * a * c],
          [alpha * b * c, b * b, alpha * a * b],
          [beta * a * c, alpha * a * b, a * a]]
S2 = np.array(np.round(S1, 6))
print("SIGMA 1 \n" + str(S1))
print("SIGMA 2 \n" + str(S2))

w1, v1 = np.linalg.eig(S1)
lambda1 = np.diag(w1)
w2, v2 = np.linalg.eig(S2)
lambda2 = np.diag(w2)

print("X1 Eigenvalues \n " + str(lambda1))
print("X2 Eigenvalues \n " + str(lambda2))
# I really don't know why it was having a size error when I did this
# individually. Figure this out later. Tomorrow perhaps?
X1, z1 = make_ZX(lambda1, M1, v1)
X2, z2 = make_ZX(lambda2, M2, v2)
print("z1 is \n" + str(z1))
print("X1 is \n" + str(X1))
# Now to plot... Perhaps make a graphing functing?
nice_plot(X1, X2, 1, 2, 'X1', 'X2')
nice_plot(X1, X2, 1, 3, 'X1', 'X2')
