import numpy as np
def sbox(input):
    list_out = [6, 4, 12, 5, 0, 7, 2, 14, 1, 15, 3, 13, 8, 10, 9, 11]
    return list_out[input]

def inverse_sbox(input):
    list_out = [6, 4, 12, 5, 0, 7, 2, 14, 1, 15, 3, 13, 8, 10, 9, 11]
    return list_out.index(input)

def encrypt(keys, message):
    ct = message
    for i in range(len(keys) - 1):
        ct ^= keys[i]
        ct = sbox(ct)
    ct ^= keys[-1]
    return ct

def permutation():
    return None



def generate_pairs(diff):
    pairs = []
    singletons = []
    for m in range(16):
        if m not in singletons:
            pairs.append((m, m ^ diff))
            singletons.append(m)
            singletons.append(m ^ diff)
    return pairs

def attack(keys):
    pairs = generate_pairs(15)
    ct_pairs = []
    inv_pairs = []
    for i in range(len(pairs)):
        pair = encrypt(keys, pairs[i][0]), encrypt(keys, pairs[i][1])
        ct_pairs.append(pair)
        #inv_pairs.append((inverse_sbox(pair[0]), inverse_sbox(pair[1])))
    ratio = [0] * 16
    for key_3 in range(16):
        for i in range(len(pairs)):
            xored_pair = key_3 ^ ct_pairs[i][0], key_3 ^ ct_pairs[i][1]
            inv_pair = (inverse_sbox(xored_pair[0]), inverse_sbox(xored_pair[1]))
            diff = inv_pair[0] ^ inv_pair[1]
            if diff == 12:
                ratio[key_3] += 1
    return ratio

def test_attack():
    good = 0
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    #keys = [i % 16, i % (16**2), i % (16**3), i % (16**4)]
                    keys = [i, j, k, l]
                    guess = np.argmax(attack(keys))
                    if guess == keys[-1]:
                        good += 1
    return good / (16**4)


keys_1 = [9, 6, 7, 12]
message_1 = 1
ct_1 = encrypt(keys_1, message_1)
#print(ct_1)

print(attack(keys_1))
#print(test_attack())
#print(2 ^ 2)

print(test_attack())
