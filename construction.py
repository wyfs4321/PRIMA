# In PRIMA, the base prefix-sum cuboid can be computed directly based on the distribution of noisy sum vector. 
# However, when the dimension of noisy sum vector is larger then 4, although it is feasible to predict the distribution 
# of multi-dimensional noisy sum vector directly by hypothesis testing, the engineering implementation of this task is complex and difficult.
# and there is no existing tools can be adopted directly for this process. Therefore, we used an estimation method in our engineering implementation,
# where we first predict the distributions of 2-d noisy sum vectors and then estimate the n-d(n>=4) distribution based on maximum entropy model.
# Notice that 1）this estimation method is not necessary and can be replaced by predicting the distribution based the hypothesis testing
# 2）this estimation method will introduce a estimation error to the result, considering the fairness, this method is also be adopted in baseline approaches.

import numpy as np
import itertools
import math
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div
from computation import Synthesizer



def max_entropy_1_to_2(A, B, n):
    l = np.concatenate((A, B), axis=0).tolist()
    p = l.copy()
    domain = [10,10]
    for i in range(l.__len__()):
        # for j in range(1, l[i].__len__()):
            # p[i][j] = l[i][j]+l[i][j-1]
        # p[i][-1] = n
        cur_m = np.sum(p[i])
        print(cur_m)
        p[i][-1] = n - cur_m + p[i][-1]
        p[i] = np.divide(p[i], n).tolist()

    result = Synthesizer.Maximum_entropy(p, None, [10,10], 1)*n
    # sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
    # p = sol['x']
    return result


def max_entropy_1_to_4(A, B, C, D, n):
    l = np.concatenate((A, B, C, D), axis=0).tolist()
    p = l.copy()
    domain = [10,10,10,10]

    for i in range(l.__len__()):
        # for j in range(1, l[i].__len__()):
            # p[i][j] = l[i][j]+l[i][j-1]
        p[i][-1]=n
        p[i] = np.divide(p[i], n).tolist()

    result = Synthesizer.Maximum_entropy(p, None, domain, 1)*n
    # sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
    # p = sol['x']
    return result


def max_entropy_2_to_4(A, B, C, D, E, F, n, domain):
    dict = {}
    dict[(0, 1)] = np.asarray(A)
    dict[(0, 2)] = np.asarray(B)
    dict[(0, 3)] = np.asarray(C)
    dict[(1, 2)] = np.asarray(D)
    dict[(1, 3)] = np.asarray(E)
    dict[(2, 3)] = np.asarray(F)


    print(n)
    for i in dict.keys():
        cur_m = np.sum(dict[i])
        print(cur_m)
        dict[i] = np.multiply(dict[i], n/cur_m)
        # dict[i][-1][-1] = n - cur_m + dict[i][-1][-1]
        dict[i] = np.divide(dict[i], n)

    result = Synthesizer.Maximum_entropy(None, dict, domain, 1) * n
    return result

# if __name__ == '__main__':
#     # l = [1,1]
    # Synthesizer.Maximum_entropy(l, None, [10,10], 2)