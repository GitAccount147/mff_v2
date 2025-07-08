import random
import math


# Factors a composite number using Pollard-Rho: modulus=p*q where p and q are primes
# Using initialization s_0 and function func (specified by constant)
def pollard_rho(modulus, func, s_0=None, constant=1):
    if s_0 is None:
        s_0 = random.randint(0, modulus)
    s = func(s_0, modulus, constant)
    s_prime = func(s, modulus, constant)
    list_sequential = [s_0, s]
    list_double = [s_0, s_prime]
    while math.gcd(modulus, s - s_prime) == 1:
        s = func(s, modulus, constant)
        s_prime = func(func(s_prime, modulus, constant), modulus, constant)
        list_sequential.append(s)
        list_double.append(s_prime)
    gcd = math.gcd(s - s_prime, modulus)
    if gcd < modulus:
        return gcd, list_sequential, list_double
    else:
        return "Failure", list_sequential, list_double


# Our choice of function: Z_N -> Z_N
def map_quad(n, modulus, constant):
    return (n ** 2 + constant) % modulus


# Gets pre-periods and periods of the sequences (induced by a function func and a starting point s_0)
def periods(s_0, modulus, func, constant=1):
    preperiod_found, preperiod, period = False, None, None
    i = 0
    val_dict = {s_0: 0}
    val_list = [s_0]
    s = s_0
    while not preperiod_found:
        s = func(s, modulus, constant)
        i += 1
        if s in val_dict.keys():
            preperiod = val_dict[s]
            period = i - preperiod
            preperiod_found = True
        val_list.append(s)
        val_dict[s] = i
    return preperiod, period, val_list


# Factors a number and the appropriate sequence of the computation
def factor(modulus, constant, s_0=2):
    p, list_sequential, list_double = pollard_rho(modulus, map_quad, s_0, constant)
    if p != "Failure":
        q = modulus // p
        print("Factorization of", modulus, "found:", p, q)
        print("List of s_i:", list_sequential)
        print("List of s_2i:", list_double)
    else:
        print("Failure.")
        print("List of s_i:", list_sequential)
        print("List of s_2i:", list_double)


# Computes the expected value (and max) of M+T (pre-period and period) over all possible starting points s_0
def expected_value(modulus, func, constant):
    summed, maximum = 0, 0

    # In Z_N:
    for s_0 in range(modulus):
        preperiod, period, _ = periods(s_0, modulus, func, constant)
        summed += preperiod + period
        if preperiod + period > maximum:
            maximum = preperiod + period

    # In Z_p:
    summed_p, maximum_p = 0, 0
    p = "Failure"
    i = 0
    while p == "Failure":
        p, _, _ = pollard_rho(modulus, func, i, constant)
        i += 1
        if i > modulus:
            print("Total failure. No choice of s_0 works.")
            p = modulus
            break
    for s_0 in range(p):
        preperiod_p, period_p, _ = periods(s_0, p, func, constant)
        summed_p += preperiod_p + period_p
        if preperiod_p + period_p > maximum_p:
            maximum_p = preperiod_p + period_p
    return summed / modulus, maximum, summed_p / p, maximum_p


# Computes the expected value (and max) of M+T (pre-period and period) over all possible starting points s_0, functions
# and different moduli
def all_moduli(moduli, constants):
    # Both over Z_N and Z_p
    list_exp, list_max, const_avg = [], [], []
    list_exp_p, list_max_p, const_avg_p = [], [], []
    for i in range(len(constants)):
        list_exp.append([])
        list_max.append([])
        list_exp_p.append([])
        list_max_p.append([])
        constant = constants[i]
        const_sum, const_sum_p = 0, 0
        for j in range(len(moduli)):
            modulus = moduli[j]
            ev, mv, ev_p, mv_p = expected_value(modulus, map_quad, constant)
            list_exp[i].append(ev)
            list_max[i].append(mv)
            const_sum += ev
            list_exp_p[i].append(ev_p)
            list_max_p[i].append(mv_p)
            const_sum_p += ev_p
        const_avg.append(const_sum / len(moduli))
        const_avg_p.append(const_sum_p / len(moduli))
    return list_exp, const_avg, list_max, list_exp_p, const_avg_p, list_max_p


# main code
def driver(moduli_list, constants_list):
    res, res_avg, res_max, res_p, res_avg_p, res_max_p = all_moduli(moduli_list, constants_list)
    for i in range(len(constants_list)):
        print("Function: f(x) = x^2 + (", constants_list[i], "):")
        print("      Average expectation over moduli_list (over Z_N): {:.2f}".format(res_avg[i]))
        print("      Maximum over moduli_list (over Z_N):", max(res_max[i]))
        # print("Values for each moduli (over Z_N):", res[i])
        print("      Average expectation over moduli_list (over Z_p): {:.2f}".format(res_avg_p[i]))
        print("      Maximum over moduli_list (over Z_p):", max(res_max_p[i]))
        # print("Values for each moduli (over Z_p):", res_p[i])


# one instance:
rsa_modulus = 3337  # assigned HW instance
quad_constant = 1  # choice of our quadratic function
initial_s = 2  # choice of our initial point
moduli_list_1 = [rsa_modulus]
constants_list_1 = [quad_constant]
moduli_list_2 = [3977, 3827, 3901, 4819, 3869, 3649, 3569, 3713, 3589, 3599, 3337, 3763, 3811, 4183, 4187]
constants_list_2 = [1, -1, 2]
factor(rsa_modulus, quad_constant, initial_s)  # factor
driver(moduli_list_1, constants_list_2)  # expectation
print("=====================================================================")

# all possibilities:
driver(moduli_list_2, constants_list_2)
