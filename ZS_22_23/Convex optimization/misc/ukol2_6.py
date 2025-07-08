import cvxpy as cp
import numpy as np

p = cp.Variable(99)

# podminky - soucet = 1, stredni hodnota a kladne hodnoty p_i
const = [cp.sum(p) == 1, np.arange(1000, 100000, 1000)@p == 38000]
for i in range(99):
    const.append(p[i] >= 0)

problem = cp.Problem(cp.Minimize(cp.sum(-cp.entr(p))), const)
problem.solve()

print(problem.value)
print(p.value)

