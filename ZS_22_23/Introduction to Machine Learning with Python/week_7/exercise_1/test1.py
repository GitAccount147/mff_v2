import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

# mine:
import sklearn.metrics

a = np.array([1, 3])
#print(a.shape)

neco = {'a': 4}
#a = neco.get('b')
#print(a)
#print(neco)
#a = 6
#print(a, neco)

ar = np.array([[2, 3], [5, 6]])
#print(ar[0] * ar[1])
#print(np.var([[1, 3], [7, 5]], axis=0))
#print(np.mean([[1, 3], [5, 5]], axis=0))
a = scipy.stats.norm(1, 1).logpdf([0, 1, 2])
print(a)

#print("hi")
for i in range(10):
    if True:
        if i == 5:
            break
        #print(i)
    #print("ha")