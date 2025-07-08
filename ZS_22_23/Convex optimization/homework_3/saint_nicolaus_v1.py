import cvxpy as cp
import numpy as np

t = cp.Variable(13)  # take in bag choice

weight = np.array([0.2, 0.6, 1.5, 2.55, 2.65, 3.15, 3.2, 3.35, 3.55, 3.95, 4.1, 4.3, 4.55])
importance = np.array([1, 2, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])
constraints = [t >= 0, t <= 1, t @ weight <= 10]

objective = cp.Minimize(- importance @ t)
# we could normalize by using c^T = (1, 0, 0) and x^T = (t, a, b)

problem = cp.Problem(objective, constraints)
problem.solve()

# results:
print("Optimal value: {:.4f}".format(problem.value))
print("Values of t:", t.value)
