import numpy as np

S_Box = [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8]
Mult = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13],
        [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2],
        [0, 4, 8, 12, 3, 7, 11, 15, 6, 2, 14, 10, 5, 1, 13, 9],
        [0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1, 9, 12, 3, 6],
        [0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4],
        [0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11],
        [0, 8, 3, 11, 6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1],
        [0, 9, 1, 8, 2, 11, 3, 10, 4, 13, 5, 12, 6, 15, 7, 14],
        [0, 10, 7, 13, 14, 4, 9, 3, 15, 5, 8, 2, 1, 11, 6, 12],
        [0, 11, 5, 14, 10, 1, 15, 4, 7, 12, 2, 9, 13, 6, 8, 3],
        [0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1, 13, 15, 3, 4, 8],
        [0, 13, 9, 4, 1, 12, 8, 5, 2, 15, 11, 6, 3, 14, 10, 7],
        [0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 8, 4, 10, 11, 5],
        [0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10]]


def mix_columns(m):  # m input, c result?
    c = [0] * 16
    c[0] = Mult[2][m[0]]^Mult[1][m[1]]^Mult[1][m[2]]^Mult[3][m[3]]  # "^" bitwise xor
    c[1] = Mult[3][m[0]]^Mult[2][m[1]]^Mult[1][m[2]]^Mult[1][m[3]]
    c[2] = Mult[1][m[0]]^Mult[3][m[1]]^Mult[2][m[2]]^Mult[1][m[3]]
    c[3] = Mult[1][m[0]]^Mult[1][m[1]]^Mult[3][m[2]]^Mult[2][m[3]]
    c[4] = Mult[2][m[4]]^Mult[1][m[5]]^Mult[1][m[6]]^Mult[3][m[7]]
    c[5] = Mult[3][m[4]]^Mult[2][m[5]]^Mult[1][m[6]]^Mult[1][m[7]]
    c[6] = Mult[1][m[4]]^Mult[3][m[5]]^Mult[2][m[6]]^Mult[1][m[7]]
    c[7] = Mult[1][m[4]]^Mult[1][m[5]]^Mult[3][m[6]]^Mult[2][m[7]]
    c[8] = Mult[2][m[8]]^Mult[1][m[9]]^Mult[1][m[10]]^Mult[3][m[11]]
    c[9] = Mult[3][m[8]]^Mult[2][m[9]]^Mult[1][m[10]]^Mult[1][m[11]]
    c[10] = Mult[1][m[8]]^Mult[3][m[9]]^Mult[2][m[10]]^Mult[1][m[11]]
    c[11] = Mult[1][m[8]]^Mult[1][m[9]]^Mult[3][m[10]]^Mult[2][m[11]]
    c[12] = Mult[2][m[12]]^Mult[1][m[13]]^Mult[1][m[14]]^Mult[3][m[15]]
    c[13] = Mult[3][m[12]]^Mult[2][m[13]]^Mult[1][m[14]]^Mult[1][m[15]]
    c[14] = Mult[1][m[12]]^Mult[3][m[13]]^Mult[2][m[14]]^Mult[1][m[15]]
    c[15] = Mult[1][m[12]]^Mult[1][m[13]]^Mult[3][m[14]]^Mult[2][m[15]]
    return c


def mix_columns_inv(m):
    c = [0] * 16
    c[0] = Mult[14][m[0]]^Mult[9][m[1]]^Mult[13][m[2]]^Mult[11][m[3]]
    c[1] = Mult[11][m[0]]^Mult[14][m[1]]^Mult[9][m[2]]^Mult[13][m[3]]
    c[2] = Mult[13][m[0]]^Mult[11][m[1]]^Mult[14][m[2]]^Mult[9][m[3]]
    c[3] = Mult[9][m[0]]^Mult[13][m[1]]^Mult[11][m[2]]^Mult[14][m[3]]
    c[4] = Mult[14][m[4]]^Mult[9][m[5]]^Mult[13][m[6]]^Mult[11][m[7]]
    c[5] = Mult[11][m[4]]^Mult[14][m[5]]^Mult[9][m[6]]^Mult[13][m[7]]
    c[6] = Mult[13][m[4]]^Mult[11][m[5]]^Mult[14][m[6]]^Mult[9][m[7]]
    c[7] = Mult[9][m[4]]^Mult[13][m[5]]^Mult[11][m[6]]^Mult[14][m[7]]
    c[8] = Mult[14][m[8]]^Mult[9][m[9]]^Mult[13][m[10]]^Mult[11][m[11]]
    c[9] = Mult[11][m[8]]^Mult[14][m[9]]^Mult[9][m[10]]^Mult[13][m[11]]
    c[10] = Mult[13][m[8]]^Mult[11][m[9]]^Mult[14][m[10]]^Mult[9][m[11]]
    c[11] = Mult[9][m[8]]^Mult[13][m[9]]^Mult[11][m[10]]^Mult[14][m[11]]
    c[12] = Mult[14][m[12]]^Mult[9][m[13]]^Mult[13][m[14]]^Mult[11][m[15]]
    c[13] = Mult[11][m[12]]^Mult[14][m[13]]^Mult[9][m[14]]^Mult[13][m[15]]
    c[14] = Mult[13][m[12]]^Mult[11][m[13]]^Mult[14][m[14]]^Mult[9][m[15]]
    c[15] = Mult[9][m[12]]^Mult[13][m[13]]^Mult[11][m[14]]^Mult[14][m[15]]
    return c


def add_key(c, key):
    res = []
    for i in range(len(c)):
        res.append(c[i] ^ key[i])
    return res


def shift_rows(c):
    res = [c[0], c[1], c[2], c[3],
           c[5], c[6], c[7], c[4],
           c[10], c[11], c[8], c[9],
           c[15], c[12], c[13], c[14]]
    return res


def shift_rows_inv(c):
    res = [c[0], c[1], c[2], c[3],
           c[7], c[4], c[5], c[6],
           c[10], c[11], c[8], c[9],
           c[13], c[14], c[15], c[12]]
    return res


def sub_bytes(c):
    res = []
    for i in range(len(c)):
        res.append(S_Box[c[i]])
    return res


def sub_bytes_inv(c):
    res = []
    for i in range(len(c)):
        res.append(S_Box.index(c[i]))
    return res


def aes_round(c, key):
    c = sub_bytes(c)
    c = shift_rows(c)
    c = mix_columns(c)
    c = add_key(c, key)
    return c


def aes_round_inv(c, key):
    c = add_key(c, key)
    c = mix_columns_inv(c)
    c = shift_rows_inv(c)
    c = sub_bytes_inv(c)
    return c


def aes(c, keys):
    c = add_key(c, keys[0])
    for i in range(1, len(keys) - 1):
        #print(i)
        c = aes_round(c, keys[i])
    c = sub_bytes(c)
    c = mix_columns(c)
    c = add_key(c, keys[-1])
    return c


def aes_inv(c, keys):
    c = add_key(c, keys[-1])
    c = mix_columns_inv(c)
    c = sub_bytes_inv(c)
    for i in range(len(keys) - 2, 0, -1):
        #print("inv:", i)
        c = aes_round_inv(c, keys[i])
    c = add_key(c, keys[0])
    return c

key_0 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
key_1 = [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0]
key_2 = [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]
key_3 = [5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6]
key_4 = [8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7]
keys_1 = [key_0, key_1, key_2, key_3, key_4]
pt_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
ct_1 = aes(pt_1, keys_1)
dec = aes_inv(ct_1, keys_1)

print("Encrypted:", pt_1)
print("Decrypted:", dec)
#print("pt, sub_bytes(pt), sub_bytes_inv(ct):", pt_1, sub_bytes(pt_1), sub_bytes_inv(sub_bytes(pt_1)), sep="\n")
#print("pt, shift_rows(pt), shift_rows_inv(ct):", pt_1, shift_rows(pt_1), shift_rows_inv(shift_rows(pt_1)), sep="\n")
#print("pt, mix_columns(pt), mix_columns_inv(ct):", pt_1, mix_columns(pt_1), mix_columns_inv(mix_columns(pt_1)), sep="\n")
#print("pt, add_key:", pt_1, add_key(pt_1, key_1), add_key(add_key(pt_1, key_1), key_1), sep="\n")

def attack(trys):
    key_guesses = []
    counts = {}
    for i1 in range(16):
        for i2 in range(16):
            for i3 in range(16):
                for i4 in range(16):
                    key_guesses.append([i1, i2, i3, i4])
                    counts[(i1, i2, i3, i4)] = 0

    #counts = [] * (16 * 16 * 16 * 16)
    for i in range(trys):
        rand_arr = np.random.randint(16, size=15)

        pts = [[i] + rand_arr.tolist() for i in range(16)]
        #print(pts)
        cts = []
        for pt in pts:
            cts.append(aes(pt, keys_1))

        for key_guess in key_guesses:
            y_sum = [0, 0, 0, 0]
            for ct in cts:
                ys = ct
                ys[0] ^= key_guess[0]
                ys[4] ^= key_guess[1]
                ys[8] ^= key_guess[2]
                ys[12] ^= key_guess[3]
            y_sum[0] ^= ys[0]
            y_sum[1] ^= ys[4]
            y_sum[2] ^= ys[8]
            y_sum[3] ^= ys[12]
            #if y_sum[0] != 0 or y_sum[1] != 0 or y_sum[2] != 0 or y_sum[3] != 0:
                #print("guess:", key_guess)
                #counts[(key_guess[0], key_guess[1], key_guess[2], key_guess[3])] = 1
            if y_sum[0] != 0:
                #print("guess:", key_guess)
                counts[(key_guess[0], key_guess[1], key_guess[2], key_guess[3])] = 1
    viable_guesses = []
    for key in counts.keys():
        if counts[key] == 0:
            viable_guesses.append(key)

    return viable_guesses


viable = attack(2)
print(viable)
