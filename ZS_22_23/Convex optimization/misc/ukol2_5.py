import cvxpy as cp
import numpy as np

a = cp.Variable()
b = cp.Variable()
s = cp.Variable()

# hledana matice
A = np.array([[1+a, -2+a-b], [-2+a-b, -1+b]])
# A = [[1, -2], [-2, -1]] + a*[[1, 1], [1, 0]] + b*[[0, -1], [-1, 1]]

f_a = np.array([[0, -1], [-1, 1]])
f_b = np.array([[1, 1], [1, 0]])
f_0 = np.array([[1, -2], [-2, -1]])


F_a = np.block([[np.zeros((2, 2)), f_a, np.zeros((2, 1))], [f_a.T, np.zeros((2, 3))], [np.zeros(5)]])
F_b = np.block([[np.zeros((2, 2)), f_b, np.zeros((2, 1))], [f_b.T, np.zeros((2, 3))], [np.zeros(5)]])
F_0 = np.block([[np.zeros((2, 2)), f_0, np.zeros((2, 1))], [f_0.T, np.zeros((2, 3))], [np.zeros(5)]])

# podminky - stavaji se z pouziti tvrzeni ze zadani (prvni 4x4 blok matice) a podminky s>=0, ktera je pripojena v
# blokem 1x1 + rozepsano do pozadovaneho tvaru (promenna krat matice)
const = [s*np.eye(5) + a*F_a + b*F_b + F_0 >> 0]

problem = cp.Problem(cp.Minimize(s), const)
problem.solve()

#print(problem.value)
print(a.value)
print(b.value)
print(np.array([[1+a.value, -2+a.value-b.value], [-2+a.value-b.value, -1+b.value]]))