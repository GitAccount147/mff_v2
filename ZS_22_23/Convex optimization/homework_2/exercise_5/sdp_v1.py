# we will use the advised equivalence, we will also decompose the original matrix into variable matrices A_a ,A_b & A_0

import cvxpy as cp
import numpy as np

t = cp.Variable()  # matrix norm parameter
a = cp.Variable()  # matrix parameter
b = cp.Variable()  # matrix parameter

# symmetric matrices
A_a = np.array([[1, 1], [1, 0]])
A_b = np.array([[0, -1], [-1, 1]])
A_0 = np.array([[1, -2], [-2, -1]])
F1 = np.eye(5)
F2 = np.block([[np.zeros((2, 2)), A_a, np.zeros((2, 1))],
               [A_a, np.zeros((2, 3))],
               [np.zeros((1, 5))]])
F3 = np.block([[np.zeros((2, 2)), A_b, np.zeros((2, 1))],
               [A_b, np.zeros((2, 3))],
               [np.zeros((1, 5))]])
G = np.block([[np.zeros((2, 2)), A_0, np.zeros((2, 1))],
              [A_0, np.zeros((2, 3))],
              [np.zeros((1, 5))]])

constraints = [t * F1 + a * F2 + b * F3 + G >> 0]
# adds the constraint 't >= 0', equivalent because of exercise (2.b)
# we could normalize by multiplying with (-1) and flipping '>>'

objective = cp.Minimize(t)
# we could normalize by using c^T = (1, 0, 0) and x^T = (t, a, b)

problem = cp.Problem(objective, constraints)
problem.solve()

# results:
print("Optimal value: {:.4f}".format(problem.value))
print("Values of t, a, b: {:.4f}, {:.4f}, {:.4f}".format(t.value, a.value, b.value))
