import random
import math
import sympy


def bin_mult(a, exp, mod):
    bin_list = [int(x) for x in bin(exp)[2:]]
    pows = []
    curr = a
    for i in range(len(bin_list)):
        curr = (curr * curr) % mod
        pows.append(curr)

    res = 1
    for i in range(len(bin_list)):
        if bin_list[i] == 1:
            res *= pows[i]
    return res


def pollard_p_minus_one(modulus, B, primes, a=None):
    if a is None:
        a = random.randint(1, modulus - 1)
    d = math.gcd(a, modulus)
    if d > 1:
        return d

    e_prime_B = 1
    for i in range(len(primes)):
        e_prime_B *= math.pow(primes[i], math.floor(math.log(B, primes[i])))
    e_prime_B = int(e_prime_B)
    #print(e_prime_B)

    #a = int(math.pow(a, e_prime_B) % modulus)  # overflows

    # too slow:
    #prod = a
    #for i in range(e_prime_B - 1):
    #    a = a * prod % modulus

    a = bin_mult(a, e_prime_B, modulus)


    d = math.gcd(a - 1, modulus)
    if d > 1:
        if d == modulus:
            return "Fail"
        else:
            return d
    stop = math.floor(math.log(B, 2))
    for i in range(stop):
        a = (a * a) % modulus
        #print(a)
        d = math.gcd(a - 1, modulus)
        if d > 1:
            if d == modulus:
                return "Fail"
            else:
                return d
    return "Fail"


def get_primes(B):
    result = list(sympy.primerange(0, B))
    return result


def find_Bs(modulus, speed_up=2, stop=35):
    stop = min(modulus, stop)
    good_Bs = []
    counts = []
    for B in range(speed_up, stop):
        primes = get_primes(B)
        viable = 0
        for a in range(1, modulus - 1):
            divisor = pollard_p_minus_one(modulus, B, primes, a)
            if divisor != "Fail":
                #print(a)
                viable += 1
        if viable != 0:
            good_Bs.append(B)
            counts.append(viable)
            #print(B)
            #break
    if len(good_Bs) != 0:
        return good_Bs[0], good_Bs, counts
    else:
        return None, good_Bs, counts

def factor_number(modulus):
    B, good_Bs, counts = find_Bs(modulus)
    print(good_Bs)
    print(counts)
    #B = 5 # change
    primes = get_primes(B)
    divisor = pollard_p_minus_one(modulus, B, primes)
    if divisor != "Fail":
        return divisor, modulus // divisor
    return "Fail"


#print(bin_mult(3, 3, 7))
my_modulus = 3589  # =37*97
print(factor_number(my_modulus))
