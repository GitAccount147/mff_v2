f = open("task_5_permutation.txt", "r")

cnt = 0
gens = []
"""
for line in f:

    #print(line)
    bits_0 = line[3:12*3:3]
    bits_1 = line[3 + 37:11 * 3 + 3 + 37:3]
    #print(bits_0)

    non_zero = 0
    for i in range(11):
        if bits_0[i] == "1":
            non_zero += 1

    if non_zero <= 2:
        cnt += 1
        #print(bits_0, bits_1)
        gens.append((bits_0, bits_1))

print(cnt)
a_i = [0] * 11

for a, b in gens:
    non_zero = 0
    for i in range(11):
        if a[i] == "1":
            non_zero += 1
    if non_zero == 1:
        for i in range(11):
            if a[i] == "1":
                break
        b = b[::-1]
        val = int(b, 2)
        a_i[i] = val
    #print(a, b)

#print(int("001", 2))
print(a_i)

a_ij = {}
for a, b in gens:
    non_zero = 0
    for i in range(11):
        if a[i] == "1":
            non_zero += 1
    indices = []
    if non_zero == 2:
        for i in range(11):
            if a[i] == "1":
                indices.append(i)

        b = b[::-1]
        val = int(b, 2)
        val = val ^ a_i[indices[0]] ^ a_i[indices[1]]
        a_ij[(indices[0], indices[1])] = val
    #print(a, b)

print(a_ij)
print(a_ij[(2, 9)])
print(1 ^ 64 ^ 544 ^ 20 ^ 128 ^ 529)
"""
dict_0 = {}
for line in f:

    #print(line)
    bits_0 = line[3:12*3:3]
    bits_1 = line[3 + 37:11 * 3 + 3 + 37:3]
    #print(bits_0)
    val_0 = int(bits_0[::-1], 2)
    val_1 = int(bits_1[::-1], 2)
    dict_0[val_0] = val_1

cnt = 0
for key in dict_0.keys():
    print(key, ": ", dict_0[key], ", ", end='', sep='')
    cnt += 1
    if cnt == 11:
        print("\n    ", end='')
        cnt = 0
#print(dict_0)

f.close()
