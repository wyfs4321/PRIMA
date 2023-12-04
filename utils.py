import itertools
import numpy as np
import math
class cartesian(object):
    def __init__(self):
        self._data_list=[]

    def add_data(self,data=[]): #添加生成笛卡尔积的数据列表
        self._data_list.append(data)

    def build(self): #计算笛卡尔积
        #print(self._data_list)
        for item in itertools.product(*self._data_list):
            print(item)

def t_add(t1,t2):
    l1=np.array(t1)
    l2=np.array(t2)
    l3=l1+l2
    t3=tuple(l3)
    return t3
def arr_extend(arr,forward):
    t1=arr.shape
    t0=[]
    for i in range(len(t1)):
        t0.append(1)
    t2=t_add(t1,t0)
    arr2=np.zeros(t2,dtype=int)

    s = cartesian()
    for i in arr.shape:
        s.add_data(range(0, i))
    for index in itertools.product(*s._data_list):
        if forward:
            arr2[t_add(index,t0)] = arr[index]
        else:
            arr2[index]=arr[index]
    return arr2
def one_steps(t):
    l=len(t)
    steps=[]
    for i in range(l):
        step=np.zeros(l,dtype=int)
        step[i]=1
        steps.append(tuple(step))
    return steps
def pfs_to_p(datas):
    #print(datas)
    l=len(datas)
    sum=datas[l-1]
    p=[]
    p.append(datas[0]/sum)
    for i in range(1,l):
        p.append((datas[i]-datas[i-1])/sum)
    #print(p)
    return p
def pfs_to_p2(datas):
    shape=datas.shape
    l=len(shape)
    t=[]
    for i in range(l):
        t.append(-1)
    t0=tuple(t)
    max_index=t_add(shape,t0)
    sum=datas[max_index]
    p=np.zeros(shape,dtype=float)
    datas2=arr_extend(datas,1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            p[i,j]=(datas2[i+1,j+1]-datas2[i,j+1]-datas2[i+1,j]+datas2[i,j])/sum
    return p
def info_entropy(p):
    n=len(p)
    e=0
    for i in range(n):
        if p[i]!=0:
            e=e+(-p[i]*math.log(p[i]))
        else:
            e=0
    return e
def get_pair(list):
    pairs=[]
    for pair in itertools.product(list,list):
        if pair[0] < pair[1]:
            pairs.append(pair)
    return pairs

def RE(sum,sum2,n):
    if sum!=0:
        return round(abs(sum-sum2)/(sum*n),10)
    else:
        return 0
def MAE(ans1,ans2,n):
    return round(abs(ans1-ans2)/n,10)

def is_Sublist(l, s):
    sub_set = False
    if s == []:
        sub_set = True
    elif s == l:
        sub_set = True
    elif len(s) > len(l):
        sub_set = False

    else:
        for i in range(len(l)):
            if l[i] == s[0]:
                n = 1
                while (n < len(s)) and (l[i + n] == s[n]):
                    n += 1

                if n == len(s):
                    sub_set = True

    return sub_set
if __name__ == '__main__':
    # s=cartesian()
    # l=[3,3]
    # for i in l:
    #     s.add_data(range(1,i+1))
    # print(s._data_list)
    # for index in itertools.product(*s._data_list):
    #     print(index)
    cartesian_obj = cartesian()

    # 添加要计算笛卡尔积的数据列表
    data_list1 = [1, 2, 3]
    data_list2 = ['A', 'B']
    cartesian_obj.add_data(data_list1)
    cartesian_obj.add_data(data_list2)

    # 计算笛卡尔积并打印结果
    cartesian_obj.build()