# Maximum entropy distribution.
import numpy as np
import itertools
import math
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div
from estimation import Synthesizer
#样本数
# n = 1000000

# 功能：通过AB两个一维边际分布和数据集大小n，根据最大熵方法进行二维边际分布的估计
# 输入：A,B——查询两个属性得到的2个一维边际分布
#       n——数据集记录大小（条数）
# 输出：result——通过最大熵估计得到的二维边际分布
def max_entropy_1_to_2(A, B, n):
    l = np.concatenate((A, B), axis=0).tolist()
    p = l.copy()
    domain = [10,10]
    # 作前缀和
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

# 功能：通过ABCD四个一维边际分布和数据集大小n，根据最大熵方法进行四维合成数据集的估计
# 输入：A,B,C,D——查询四个属性得到的4个一维边际分布
#       n——数据集记录大小（条数）
# 输出：result——通过最大熵估计得到的四维合成数据集
def max_entropy_1_to_4(A, B, C, D, n):
    l = np.concatenate((A, B, C, D), axis=0).tolist()
    p = l.copy()
    domain = [10,10,10,10]
    # 作前缀和
    for i in range(l.__len__()):
        # for j in range(1, l[i].__len__()):
            # p[i][j] = l[i][j]+l[i][j-1]
        p[i][-1]=n
        p[i] = np.divide(p[i], n).tolist()

    result = Synthesizer.Maximum_entropy(p, None, domain, 1)*n
    # sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
    # p = sol['x']
    return result

# 功能：通过ABCDEF六个二维边际分布和数据集大小n，根据最大熵方法进行四维合成数据集的估计
# 输入：A,B,C,D,E,F——查询四个属性得到的C(4,2)=6个二维边际分布
#       n——标准四维分布的求和值
#       domain——目标四维分布的大小
# 输出：result——通过最大熵估计得到的四维合成数据集
def max_entropy_2_to_4(A, B, C, D, E, F, n, domain):
    dict = {}
    dict[(0, 1)] = np.asarray(A)
    dict[(0, 2)] = np.asarray(B)
    dict[(0, 3)] = np.asarray(C)
    dict[(1, 2)] = np.asarray(D)
    dict[(1, 3)] = np.asarray(E)
    dict[(2, 3)] = np.asarray(F)

    # 作前缀和
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