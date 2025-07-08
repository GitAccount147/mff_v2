import numpy as np
def sbox(input):
    list_out = [6, 4, 12, 5, 0, 7, 2, 14, 1, 15, 3, 13, 8, 10, 9, 11]
    return list_out[input]

def inverse_sbox(input):
    list_out = [6, 4, 12, 5, 0, 7, 2, 14, 1, 15, 3, 13, 8, 10, 9, 11]
    return list_out.index(input)


def sbox_layer(input):
    out = []
    for i in range(4):
        #print((input >> (i * 4) % 16))
        out.append(sbox(((input >> (i * 4)) % 16)))
    res = 0
    for i in range(4):
        res += out[i] * (16 ** i)
    return res

def sbox_layer_inverse(input):
    out = []
    for i in range(4):
        #print(((input >> (i * 4)) % 16))
        out.append(inverse_sbox(((input >> (i * 4)) % 16)))
    res = 0
    for i in range(4):
        res += out[i] * (16 ** i)
    return res


def permutation(input):
    inp = input
    perm = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    arr = [0] * 16
    for i in range(16):
        if inp % 2 == 1:
            arr[perm[15 - i]] = 1
        inp = inp >> 1
    return int("".join(str(x) for x in arr), 2)

def inverse_permutation(input):
    inp = input
    perm = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    arr = [0] * 16
    for i in range(16):
        if inp % 2 == 1:
            arr[perm.index(15 - i)] = 1
        inp = inp >> 1
    return int("".join(str(x) for x in arr), 2)


def encrypt(keys, message):
    ct = message
    for i in range(len(keys) - 2):
        ct ^= keys[i]
        ct = sbox_layer(ct)
        ct = permutation(ct)
    ct ^= keys[-2]
    ct = sbox_layer(ct)
    ct ^= keys[-1]
    return ct

def decrypt(keys, ct):
    pt = ct
    pt ^= keys[-1]
    pt = sbox_layer_inverse(pt)
    pt ^= keys[-2]
    for i in range(len(keys) - 3, -1, -1):
        pt = inverse_permutation(pt)
        pt = sbox_layer_inverse(pt)
        pt ^= keys[i]
    return pt


def generate_pairs(diff, number):
    df = "".join("0"*(4-len(bin(x)[2:]))+str(bin(x)[2:]) for x in diff)
    #a=  int("".join(str(bin(x)) for x in diff), 2)
    pairs = []
    singletons = []
    curr = 0
    for m in range(2**16):
        if m not in singletons:
            pairs.append((m, m ^ diff))
            singletons.append(m)
            singletons.append(m ^ diff)
            curr += 1
        if curr == number:
            break
    return pairs

def attack(keys):
    pairs = generate_pairs(2)
    ct_pairs = []
    inv_pairs = []
    for i in range(len(pairs)):
        pair = encrypt(keys, pairs[i][0]), encrypt(keys, pairs[i][1])
        ct_pairs.append(pair)
        #inv_pairs.append((inverse_sbox(pair[0]), inverse_sbox(pair[1])))
    ratio = [0] * 16
    for last_key in range(16):
        for i in range(len(pairs)):
            xored_pair = last_key ^ ct_pairs[i][0], last_key ^ ct_pairs[i][1]
            inv_pair = (inverse_sbox(xored_pair[0]), inverse_sbox(xored_pair[1]))
            diff = inv_pair[0] ^ inv_pair[1]
            if diff == 12:
                ratio[key_3] += 1
    return ratio

#print(7 >> 2)
#print(int("".join(str(x) for x in [1, 0, 1, 1]),2))

#print(decrypt([1,2,3,4,5,6], encrypt([1,2,3,4,5,6],14)))

print(["0"*(4-len(bin(x)[2:]))+str(bin(x)[2:]) for x in [2]])


