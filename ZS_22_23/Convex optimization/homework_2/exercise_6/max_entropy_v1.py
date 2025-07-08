import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# solves an instance of the problem
def optimization(n, scale, avg):
    avg_salary = avg * scale
    p = cp.Variable(n)
    salary_values = np.arange(scale, (n + 1) * scale, scale)

    constraints = [  # we will assume that none of the p_i is equal to 0, otherwise we would have to take care
        # of the values 0*log(0) in the entropy
        p >= np.zeros(n),
        cp.sum(p) == 1,
        salary_values @ p == avg_salary  # calculate the expected value
    ]
    objective = cp.Minimize(cp.sum(- cp.entr(p)))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return prob.value, p.value


# parameters from the main exercise:
scale = 1000
n = 99
average_salary = 38

# plots solutions for other parameters
avgs = np.arange(30, 50, 2)
for i in range(len(avgs)):
    res = optimization(n, scale, avgs[i])
    col = [0, 0.5, 0.1 + i / len(avgs), 0.5]
    plt.plot(np.arange(scale, (n + 1) * scale, scale), res[1], color=col)
    plt.plot([avgs[i] * scale], [res[1][round(avgs[i])]], '-o', color=col, markersize=4)
plt.axvline(x=(38 * scale), color='orange', label='Average salary')

# the main exercise:
main_res = optimization(n, scale, average_salary)
plt.plot(np.arange(scale, (n + 1) * scale, scale), main_res[1], color=[1, 0, 0], label='Optimal for avg_sal')

# results:
print("Optimal value: {:.4f}".format(main_res[0]))
print("Probabilities p:\n", main_res[1])
legend = plt.legend()
plt.show()
