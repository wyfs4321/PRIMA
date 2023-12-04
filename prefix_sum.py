import numpy as np
import pandas as pd
import pprint
import math
from scipy.stats import norm,shapiro,kstest,normaltest,anderson
import scipy.stats as stats
import scipy
import count_query
import auto_trip
import argparse
import max_entropy
import data_preprocess
import time
import datetime
# import fitter as fitter
# from fitter import Fitter
from distfit import distfit
# import matplotlib
import matplotlib.pyplot as plt

data_dir = "./data/"
max_times_noise = 1000
n = 1000001
p = 1e-5
# 参数
alpha = 1.0
beta = 0.0
max_f = 0.01
r2t_beta = 0.1
scale = 1

def str2bool(str):
    return True if str.lower()=='true' else False

parser = argparse.ArgumentParser()
parser.add_argument('--enable_external', action='store_true', help='读取已经保存的四维数据')
parser.add_argument('--data_path', default='./virtual.dat', help='四维数据路径')
parser.add_argument('--column_1', default=0, help='四维数据列1')
parser.add_argument('--column_2', default=1, help='四维数据列2')
parser.add_argument('--column_3', default=2, help='四维数据列3')
parser.add_argument('--column_4', default=3, help='四维数据列4')
parser.add_argument('--target_column',default=12,type=int, help='目标求和列')
parser.add_argument('--alpha',default=1, type=float, help='alpha值')
parser.add_argument('--beta',default=0, type=float, help='beta值')
parser.add_argument('--maxf',default=0.01, type=float, help='max_f值')
parser.add_argument('--n',default=100000, type=int, help='数据量')
parser.add_argument('--disable_maxentropy', type=str2bool,default="False", help='不进行最大熵估计')
parser.add_argument('--disable_prefixSum', action='store_true', help='不进行前缀和估计')
parser.add_argument('--clip_method', default='R2TP', help='裁剪方法')
parser.add_argument('--epsilon', default=3,type=float, help='隐私预算大小')
parser.add_argument('--value_dimen_origin', default=30,type=int, help='原始数据值域')
parser.add_argument('--dataset', default='IPUMS', help='数据集')



opt, unknown = parser.parse_known_args()
print(opt)
colum_list = [opt.column_1, opt.column_2, opt.column_3, opt.column_4]
target_column = opt.target_column
alpha = opt.alpha
beta = opt.beta
max_f = opt.maxf
n = opt.n
clip_method = opt.clip_method
epsilon = opt.epsilon

# 数据集的原始值域
value_dimen_origin = opt.value_dimen_origin
value_dimen_process = 10
# 处理前的数据文件相对路径
source_filename = 'virtual/'+ opt.dataset +'_10-'+ str(value_dimen_origin)+ '.csv'
# 处理完成的数据文件相对路径
process_filename = 'processed/'+ opt.dataset+ '_10-'+ str(value_dimen_origin)+ 'to10.csv'

# 功能：读取数据集
# 输出：df——返回读取的数据集的dataFrame形式
def read_data_virtual():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(data_dir + process_filename, header=None, encoding='utf-8')
    df = df[0: n]
    pprint.pprint(df.head(5))
    print(df.__len__())
    return df

# 功能：数据值域压缩预处理
# 无输入输出，只是函数的封装
def data_preprocess_fun():
    print("数据预处理")
    data_high = data_preprocess.read_data_high_level(data_dir + source_filename)
    data_low = data_preprocess.process_high_to_low(data_high, value_dimen_origin, value_dimen_process)
    data_low.to_csv(data_dir + process_filename, header=0, index=0)
    return


# 功能：从df数据集中查询属性为a和属性为b的列，返回实际二维边际分布D_origin和加噪后的二维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a,b——需要统计的属性名称
# 输出：D_origin——查询得到的实际二维边际分布
#       D_noise——加噪后的二维边际分布
def query_generate_dimension_two(df, a, b):
    D_origin = np.zeros((value_dimen_process, value_dimen_process))
    for i in range(value_dimen_process):
        for j in range(value_dimen_process):
            D_origin[i][j] = df.loc[(df[a] == i) & (df[b] == j), target_column].sum()
    # 灵敏度
    f = df[target_column].max()
    D_noise = D_origin + np.random.laplace(0, f / epsilon, (value_dimen_process, value_dimen_process))
    # print(D_origin)
    # print(D_noise)
    return D_origin,D_noise

# 功能：从df数据集中查询属性为a和属性为b的列，返回实际二维边际分布D_origin和R2T查询后的二维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a,b——需要统计的属性名称
# 输出：D_origin——查询得到的实际二维边际分布
#       D_noise——加噪后的二维边际分布
def query_generate_dimension_two_with_R2T(df, a, b):
    D_origin = np.zeros((value_dimen_process, value_dimen_process))
    D_noise = np.zeros((value_dimen_process, value_dimen_process))
    m_domain = np.int(df[target_column].max() + 1)

    for i in range(value_dimen_process):
        for j in range(value_dimen_process):
            D_origin[i][j] = df.loc[(df[a] == i) & (df[b] == j), target_column].sum()
            D_noise[i][j] = R2T(df, m_domain, a, b, i,j,epsilon,r2t_beta)
    return D_origin,D_noise

# 功能：从df数据集中查询属性为abcd的列，返回实际四维边际分布D_origin和R2T查询后的四维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a,b,c,d——需要统计的属性名称
# 输出：D_origin——查询得到的实际四维边际分布
#       D_noise——加噪后的四维边际分布
def query_generate_dimension_four_with_R2T(df, a, b, c, d):
    D_origin = np.zeros((value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process))
    D_noise = np.zeros((value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process))
    m_domain = np.int(df[target_column].max() + 1)

    for i in range(value_dimen_process):
        for j in range(value_dimen_process):
            for k in range(value_dimen_process):
                for l in range(value_dimen_process):
                    D_origin[i][j][k][l] = df.loc[(df[a] == i) & (df[b] == j) & (df[c] == k) & (df[d]==l), target_column].sum()
                    D_noise[i][j][k][l] = R2T_four(df, m_domain, a, b,c,d,i,j,k,l,epsilon,r2t_beta)
    return D_origin,D_noise

# 功能：从df数据集中查询属性为a、b、c、d的列，返回实际四维边际分布D_origin和加噪后的四维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a,b,c,d——需要统计的属性名称
# 输出：D_origin——查询得到的实际四维边际分布
#       D_noise——加噪后的四维边际分布
def query_generate_dimension_four(df, a, b, c, d):
    # len_m = df[a].unique().__len__()
    # len_n = df[b].unique().__len__()
    # len_o = df[c].unique().__len__()
    # len_p = df[d].unique().__len__()
    D_origin = np.zeros((value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process))
    for i in range(value_dimen_process):
        for j in range(value_dimen_process):
            for k in range(value_dimen_process):
                for l in range(value_dimen_process):
                    D_origin[i][j][k][l] = df.loc[(df[a]==i)
                                                  & (df[b]==j)
                                                  & (df[c]==k)
                                                  & (df[d]==l), target_column].sum()

    # 灵敏度
    f = df[target_column].max()
    print("灵敏度{}".format(f))
    D_noise = D_origin + np.random.laplace(0, f / epsilon, (value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process))
    return D_origin, D_noise

# 功能：对目标列进行适当的裁剪，返回裁剪后的数据集
# 输入：df——需要进行遍历的数据集
# 输出：df——经过裁剪的数据集
def clip_target_column(df):
    tmin, tmax = auto_trip.get_min_max_from_data(df, target_column)
    print("目标数据值范围:["+str(tmin)+","+ str(tmax)+"]")
    m, t = auto_trip.get_avg_std_from_data(df, target_column)

    # 自适应获取裁剪阈值
    tao = auto_trip.auto_clip(df, target_column, m, tmax, epsilon)
    print("自适应裁剪半径:" + str(tao) + ",裁剪范围：[" + str(max(m - tao, tmin)) + "," + str(min(m + tao, tmax)) + "]")

    # 稀疏向量技术无偏处理
    # 如果单边裁剪
    if m - tao <= tmin and m + tao < tmax:
        best_min_range = auto_trip.no_bias(df, target_column, tmin, m + tao)
        df = auto_trip.data_trip(df, target_column, best_min_range, m + tao)
        print("无偏处理后的裁剪范围:(" + str(best_min_range) + "," + str(m + tao) + ")")
    elif m + tao >= tmax:
        print("决策为不裁剪，即范围是：(" + str(tmin) + "," + str(tmax) + ")")
    else:
        df = auto_trip.data_trip(df, target_column, m - tao, m + tao)
        print("决策为双边裁剪，即范围是：（" + str(m - tao) + "," + str(m + tao) + ")")
    return df

# SVT所需的差异查询流
# 输入：数据表、各个维度属性的值域（默认一样）、参与的属性、隐私预算
# 输出：差异查询流的综合
def difference_query(df,domain,Da,Db,b,epsilon_i):
    summation=0
    for i in range(domain):
        for j in range(domain):
            summation+=df.loc[(df[Da] == i) & (df[Db] == j), target_column].sum() + np.random.laplace(0, b / epsilon_i)
            # summation+=df.loc[(df[Da] == i) & (df[Db] == j), target_column].sum()

    return summation
# 稀疏向量技术确定裁剪上限
# 输入：数据表、一个需要人为预设的判断阈值、待求和的度量值的值域、各个维度属性的值域（默认一样）、隐私预算、参与的属性
# 输出：一个最优的阈值
def clip_target_column_SVT(df,threshold,m_domain,d_domain,epsilon,Da=None,Db=None,Dc=None,Dd=None):
    bs = range(1, m_domain, int(m_domain/100))
    best = 0
    epsilon_i = epsilon / len(bs)
    for b in bs:
        tmp_df = auto_trip.data_trip(df, target_column, 0, b)
        r = difference_query(tmp_df, d_domain,Da,Db,b,epsilon_i)
        # if the new answer is pretty close to the old answer, stop
        if abs(r - best) <= threshold:
            return b
        # otherwise update the "best" answer to be the current one
        else:
            best = r
    return bs[-1]
# R2T算法
# 输入：数据表、待求和的度量值的值域、参与的属性、参与的属性的限制范围、隐私预算、一个需要人为预设的错误率、
def R2T(df,m_domain,Da,Db,i,j,epsilon,beta):
    bs=[2**k for k in range(1, int(math.log(m_domain,2)))]
    r=[]
    epsilon_i=epsilon/len(bs)
    for b in bs:
        tmp_df=auto_trip.data_trip(df, target_column, 0, b)
        r.append(tmp_df.loc[(tmp_df[Da] == i) & (tmp_df[Db] == j), target_column].sum() + np.random.laplace(0, b / epsilon_i)-math.log(m_domain,2)*math.log(math.log(m_domain,2)/beta, math.e)*b/epsilon_i)
    return max(r)

# R2T算法
# 输入：数据表、待求和的度量值的值域、参与的属性、参与的属性的限制范围、隐私预算、一个需要人为预设的错误率、
def R2T_four(df,m_domain,Da,Db,Dc,Dd,i,j,k,l,epsilon,beta):
    bs=[2**k for k in range(1, int(math.log(m_domain,2)))]
    r=[]
    epsilon_i=epsilon/len(bs)
    for b in bs:
        tmp_df=auto_trip.data_trip(df, target_column, 0, b)
        r.append(tmp_df.loc[(tmp_df[Da] == i) & (tmp_df[Db] == j) & (tmp_df[Dc] == k) & (tmp_df[Dd]==l), target_column].sum() + np.random.laplace(0, b / epsilon_i)-math.log(m_domain,2)*math.log(math.log(m_domain,2)/beta, math.e)*b/epsilon_i)
    return max(r)

def fitter_judge(table):
    f = Fitter(table,distributions = ['uniform','norm'])  # 创建Fitter类
    f.fit()  # 调用fit函数拟合分布
    result = f.summary(10)  # 输出拟合结果
    pprint.pprint(result)
    plt.show()

    # f.plot_pdf(names=None, Nbest=5, lw=2, method='sumsquare_error')
    # plt.show()
    return f.df_errors,f.get_best()

def distfit_judge(table):
    dist = distfit(distr=['norm', 'uniform'], bins=len(table.value_counts()))
    if (len(table.value_counts())==1):
        print("skip")
        return
    result = dist.fit_transform(table)
    # print(dist.summary)
    # dist.plot()
    # plt.show()
    return result

def normal_judge_1(table):
    stat, c_list, p_list = anderson(table, dist='norm')
    stat, c_list, p_list = anderson(table, dist='expon')
    stat, c_list, p_list = anderson(table, dist='logistic')
    stat, c_list, p_list = anderson(table, dist='gumbel')
    stat, c_list, p_list = anderson(table, dist='extreme1')

    if c_list.any()>stat:
        return True
    else:
        return False

def normal_judge_2(table):
    ks = stats.t.fit(table)
    df = ks[0]
    loc = ks[1]
    scale = ks[2]
    ks2 = stats.t.rvs(df=df, loc=loc, scale=scale, size=table.shape)
    stat, p = stats.ttest_ind(table, ks2)
    if p>0.05:
        return ks2
    else:
        return None

def normal_judge_3(table):
    ks = stats.chi2.fit(table)
    df = ks[0]
    loc = ks[1]
    scale = ks[2]
    ks2 = stats.chi2.rvs(df=df, loc=loc, scale=scale, size=table.shape)
    stat, p = stats.ks_2samp(table, ks2)
    if p>0.05:
        return ks2
    else:
        return None

def normal_judge(table):
    stat, p = normaltest(table,axis=None)

    if p>0.05:
        return True
    else:
        return False

# 一维正态分布概率密度函数
# 输入：x——函数变量
# 输出：标准一维正态分布概率密度函数值
def f(x):
    return 1.0 / pow(2*math.pi,0.5) * pow(math.e,-(x*x/2))

# 二维正态分布概率密度函数
# 输入：i,j——数据所处行列
#       u1,u2——两个维度上的数据均值
#       sigma1,sigma2——两个维度上的数据标准差
# 输出：x——(i,j)位置的数据在二维正态分布概率密度函数上的映射值
def f2(i,j, u1, u2, sigma1, sigma2):
    x = scipy.stats.multivariate_normal(mean=np.array([0, 0]),cov=np.array([[1,0],[0,1]])).pdf(
        x=np.array([(j - u2 - beta * sigma2) / sigma2 * (1/u2), (i - u1 - beta * sigma1) / sigma1 * (1/u1)]))
    # return max(max_f, x)
    return x

# 二维均匀分布概率密度函数
# 输入：i,j——数据所处行列
#       u1,u2——两个维度上的数据均值
#       sigma1,sigma2——两个维度上的数据标准差
# 输出：x——(i,j)位置的数据在二维均匀分布概率密度函数上的映射值
def f1(i,j, u1, u2, sigma1, sigma2):
    return 1

# 二维正态分布累积函数
# 输入：i,j——数据所处行列
#       u1,u2——两个维度上的数据均值
#       sigma1,sigma2——两个维度上的数据标准差
# 输出：x——(i,j)位置的数据在二维正态分布分布累积函数上的映射值
def F2(i, j, u1, u2, sigma1, sigma2):
    x = scipy.stats.multivariate_normal(mean=np.array([0,0]),cov=np.array([[1,0],[0,1]])).cdf(
        x=np.array([(j - u2 - beta * sigma2) / sigma2 * (1/u2), (i - u1 - beta * sigma1) / sigma1 * (1/u1)]))

    return x

# 二维均匀分布累积函数
# 输入：i,j——数据所处行列
#       u1,u2——两个维度上的数据均值
#       sigma1,sigma2——两个维度上的数据标准差
# 输出：x——(i,j)位置的数据在二维正均匀分布分布累积函数上的映射值
def F1(i, j, u1, u2, sigma1, sigma2):
    return (i+1)*(j+1)

# 一维正态分布累积函数
# 输入：x——函数变量
# 输出：标准一维正态分布累积函数值
def P(x):
    return norm.cdf(x)  # 累计密度函数

# 一维前缀和估计函数
# 输入：matrix——需要进行前缀和估计的数据（已加噪）
# 输出：extimate_matrix——粗略估计得到的前缀和形式的矩阵
def estimate_sum_single_dimens(matrix):
    u = getU(matrix)
    sigma = getSigma(matrix, u)
    # print("u:"+str(u)+"sigma:"+str(sigma))
    extimate_matrix = matrix.copy()
    for i in range(0,extimate_matrix.__len__()):
        new_x = (i-u)/sigma
        # print("x="+str(new_x)+"，f(x)="+str(f(new_x))+"，P(x)="+str(P(new_x)))
        extimate_matrix[i] = matrix[i]/f(new_x)*P(new_x)
    # print(extimate_matrix)
    return extimate_matrix

# 分布均值寻找函数
# 输入：matrix——需要寻找分布均值的一维数据
# 输出：该组数据的分布均值
def getU(matrix):
    mean = 0.0
    for i in range(matrix.__len__()):
        mean = mean + matrix[i]*i
    return mean/np.sum(matrix)

# 分布标准差寻找函数
# 输入：matrix——需要寻找分布标准差的一维数据
# 输出：该组数据的分布标准差
def getSigma(matrix, U):
    sigma = 0.0
    for i in range(matrix.__len__()):
        sigma = sigma + math.pow((U-i),2)* matrix[i]
    return math.sqrt(sigma/np.sum(matrix))

# 二维前缀和估计函数
# 输入：matrix——需要进行前缀和估计的数据（已加噪）
# 输出：extimate_matrix——粗略估计得到的前缀和形式的矩阵
def estimate_sum_two_dimens(matrix, b1, b2):
    A = np.sum(matrix, axis=1)
    B = np.sum(matrix, axis=0)
    # print(A,B)
    u1 = getU(A)
    sigma1 = getSigma(A,u1)
    u2 = getU(B)
    sigma2 = getSigma(B,u2)

    if (b1['model']['name']=='norm'):
        is_uniform = False
    else:
        is_uniform = True
    # print("u1:"+str(u1)+",sigma1:"+str(sigma1))
    # print("u2:"+str(u2)+",sigma2:"+str(sigma2))
    extimate_matrix = matrix.copy()
    for i in range(0,extimate_matrix.__len__()):
        for j in range(0, extimate_matrix[i].__len__()):
            # f = f1(i, j, u1, u2, sigma1, sigma2) if is_uniform else f2(i, j, u1, u2, sigma1, sigma2)
            # P = F1(i, j, u1, u2, sigma1, sigma2) if is_uniform else F2(i, j, u1, u2, sigma1, sigma2)
            f = f1(i, j, u1, u2, sigma1, sigma2) if is_uniform else f2(i, j, b1['model']['loc'], b2['model']['loc'], b1['model']['scale'], b2['model']['scale'])
            P = F1(i, j, u1, u2, sigma1, sigma2) if is_uniform else F2(i, j, b1['model']['loc'], b2['model']['loc'], b1['model']['scale'], b2['model']['scale'])
            # f = max(max_f, f)
            # print("f="+str(f)+",P="+str(P)+",P/f="+str(P/f)+",matrix="+str(matrix[i][j]))
            extimate_matrix[i][j] = matrix[i][j]*P/(f*alpha)
    # print(extimate_matrix)
    return extimate_matrix

# 一维前缀和估计函数的辅助函数
# 输入：matrix——需要进行计算的矩阵
# 输出：new_matrix——用于前缀和估计的辅助数组m
def compute_mij(matrix):
    new_matrix = np.zeros((matrix.__len__(),matrix.__len__()))
    for i in range(matrix.__len__()):
        for j in range(matrix.__len__()):
            for k in range(i,j+1):
                new_matrix[i,j]=new_matrix[i,j]+matrix[k]/(j-i+1)
    # print(new_matrix)
    return new_matrix

# 一维后处理函数
# 输入：matrix——需要进行后处理纠正的前缀和形式的矩阵
# 输出：new_matrix——数据一致性处理完成后的前缀和形式的矩阵
def post_process(matrix):
    new_matrix = np.zeros((1,matrix.__len__()))

    matrix_m = compute_mij(matrix)
    for k in range(matrix.__len__()):
        min_v = 10e9
        for j in range(k, matrix.__len__()):
            max_v = 0
            for i in range(j+1):
                max_v = max(max_v, matrix_m[i,j])
            min_v = min(max_v, min_v)
        new_matrix[0,k] = min_v
    # print(new_matrix)
    return new_matrix

# 二维后处理函数
# 输入：matrix——需要进行后处理纠正的前缀和形式的矩阵
# 输出：result_matrix——数据一致性处理完成后的前缀和形式的矩阵
def post_process_double(matrix):
    result_matrix = matrix.copy()
    result_matrix = np.maximum(result_matrix, 0)
    for i in range(matrix.__len__()):
        for j in range(matrix[i].__len__()):
            if (j+1<matrix[i].__len__() and result_matrix[i][j+1]<result_matrix[i][j]):
                result_matrix[i][j+1]=result_matrix[i][j]
            if (i+1<matrix.__len__() and result_matrix[i+1][j]<result_matrix[i][j]):
                result_matrix[i+1][j]=result_matrix[i][j]
            if (i+1<matrix.__len__() and j+1<matrix[i].__len__() and result_matrix[i+1][j+1]<result_matrix[i+1][j]+result_matrix[i][j+1]-result_matrix[i][j]):
                result_matrix[i + 1][j + 1] = result_matrix[i + 1][j] + result_matrix[i][j + 1] - result_matrix[i][j]
    # print(result_matrix)
    return result_matrix

def normalize(estimate, origin):
    p = estimate.sum()/origin.sum()
    estimate = np.divide(estimate, p)
    return estimate

# 前缀和计算函数
# 输入：matrix——需要进行前缀和转化的原始矩阵
# 输出：result_matrix——前缀和形式的矩阵
def get_prefix_sum(matrix):
    result_matrix = matrix.copy()
    for i in range(result_matrix.__len__()):
        for j in range(result_matrix[i].__len__()):
            if (i==0 and j!=0):
                result_matrix[i][j] = result_matrix[i][j-1]+matrix[i][j]
            elif(i!=0 and j==0):
                result_matrix[i][j] = result_matrix[i-1][j]+matrix[i][j]
            elif(i==0 and j==0):
                continue
            else:
                result_matrix[i][j] = result_matrix[i-1][j]+result_matrix[i][j-1]-result_matrix[i-1][j-1]+matrix[i][j]
    # print(result_matrix)
    return result_matrix

# 通过前缀和得到原始数据的转化函数
# 输入：matrix——前缀和形式的矩阵
# 输出：result_matrix——从前缀和形式转化而成的原始数据形式的矩阵
def get_original_matrix(matrix):
    result_matrix = matrix.copy()
    for i in range(result_matrix.__len__()):
        for j in range(result_matrix[i].__len__()):
            if (i==0 and j!=0):
                result_matrix[i][j] = matrix[i][j]-matrix[i][j-1]
            elif(i!=0 and j==0):
                result_matrix[i][j] = matrix[i][j]-matrix[i-1][j]
            elif(i==0 and j==0):
                continue
            else:
                result_matrix[i][j] = matrix[i][j]+matrix[i-1][j-1]-matrix[i][j-1]-matrix[i-1][j]
    result_matrix = np.maximum(result_matrix, 0)
    # print(result_matrix)
    return result_matrix

# 计算原始查询、噪声查询、估计查询
# 输入：data——需要进行遍历的数据集
#       A,B——需要统计的属性名称
# 输出：origin——查询得到的实际二维边际分布
#       noise——加噪后的二维边际分布
#       estimate——经过前缀和估计后得到的二维边际分布
#                   分为三步：前缀和估计、一致性处理（后处理纠正）
#                   和原始数据转变
def query_noise_with_estimate(data, A, B, clip_method=None):
    if clip_method=="R2TP":
        origin, noise = query_generate_dimension_two_with_R2T(data, A, B)
    else:
        origin, noise = query_generate_dimension_two(data, A, B)

    if opt.disable_prefixSum:
        return origin, noise, noise

    # 加噪数据应该得出的前缀和
    prefix_sum = get_prefix_sum(origin)
    prefix_origin = get_original_matrix(prefix_sum)

    noise = np.maximum(noise, 0)

    estimate = estimate_sum_two_dimens(noise, dict_best[A], dict_best[B])
    estimate = post_process_double(estimate)
    # 归一化
    # estimate = normalize(estimate, prefix_sum)
    estimate = get_original_matrix(estimate)
    # 归一化
    # estimate = normalize(estimate, origin)

    return origin, noise, estimate

# 计算相对误差
# 输入：A,B——两个需要进行误差计算的矩阵
#       p——误差计算时所需参数
def compute_error(A, B, p, q=None):
    start = time.perf_counter()
    if q is None:
        err, eval_num = count_query.compute_MNAE(A, B, 1/value_dimen_origin, p, value_dimen_origin, B.sum(), method=1)
    else:
        err, eval_num = count_query.compute_MNAE(A, B, q, p, value_dimen_origin, B.sum(),method=1)

    # 查询次数
    query_time = value_dimen_origin
    print("平均误差：" + str(err.sum()/eval_num))
    print("目前耗时：" + str(time.perf_counter() - start) + "s")
    print("查询次数：" + str(pow(query_time, 4)))
    return err, eval_num

# 计算随机误差
# 输入：A,B——两个需要进行误差计算的矩阵
#       p——误差计算时所需参数
def compute_random_error(A, B, n_list):
    start = time.perf_counter()

    err = A-B
    err_use = np.array(err).flatten()
    print("目前耗时：" + str(time.perf_counter() - start) + "s")

    for union in n_list:
        select = np.random.choice(err_use, size=union, replace=False)
        # 查询次数
        print("平均误差：" + str(select.mean()))
        print("查询次数：" + str(union))
        # 输出实验结果
        with open("clip-result.txt", 'a', encoding='utf-8') as f_six:
            f_six.write("dataset={},epsilon={},dom(D)={},target(M)={},n={},clip_method={},union={},result={}\n"
                        .format(opt.dataset, epsilon, value_dimen_origin, target_column, n, clip_method, union, select.mean()))
    return select, n

if __name__ == "__main__":
    start = time.perf_counter()

    # 数据预处理
    data_preprocess_fun()
    start = time.perf_counter()

    # 读取数据
    df = read_data_virtual()
    print("读取数据时间：" + str(time.perf_counter() - start) + "s")
    start = time.perf_counter()

    D_origin, D_noise = query_generate_dimension_four(df, colum_list[0], colum_list[1], colum_list[2], colum_list[3])
    print("查询四维实际分布大小,size:" + str(D_origin.size))
    print("查询四维实际分布时间：" + str(time.perf_counter() - start) + "s")
    start = time.perf_counter()

    if opt.disable_maxentropy:
        # 不进行最大熵估计

        # 裁切
        # 对目标求和列作自适应裁剪
        print("开始裁剪")
        if clip_method == "April":
            # 方法一
            df = clip_target_column(df)
            clip_origin, clip_noise = query_generate_dimension_four(df, colum_list[0], colum_list[1], colum_list[2], colum_list[3])
        elif clip_method == "SVTP":
            # 方法二
            m_domain = np.int(df[target_column].max() + 1)
            max_range = clip_target_column_SVT(df, m_domain * 30, m_domain, value_dimen_process, epsilon, colum_list[0],
                                               colum_list[1], colum_list[2], colum_list[3])
            df = auto_trip.data_trip(df, target_column, 0, max_range)
            print("处理后的裁剪范围:(" + str(0) + "," + str(max_range) + ")")
            clip_origin, clip_noise = query_generate_dimension_four(df, colum_list[0], colum_list[1], colum_list[2], colum_list[3])
        elif clip_method=='R2TP':
            clip_origin, clip_noise = query_generate_dimension_four_with_R2T(df, colum_list[0], colum_list[1], colum_list[2], colum_list[3])

        print("裁剪数据时间：" + str(time.perf_counter() - start) + "s")
        start = time.perf_counter()
        # 进行维度扩展
        D_origin = count_query.expand_data(D_origin, value_dimen_origin, value_dimen_process)
        D_noise = count_query.expand_data(D_noise, value_dimen_origin, value_dimen_process)
        if clip_method=="noop":
            clip_noise = D_noise
        else:
            clip_noise = count_query.expand_data(clip_noise, value_dimen_origin, value_dimen_process)

        # 计算平均误差
        # err, eval_num = compute_error(D_noise, D_origin, p)
        err, eval_num = compute_random_error(clip_noise, D_origin, [10,100,1000,10000])
        # for q in range(1,11):
        #     compute_error(D_noise,D_origin, p, q/10)


    else:
        if opt.enable_external:
            # 通过磁盘存储的已合成数据集读取估计四维合成数据集
            estimate = np.load(opt.data_path, allow_pickle=True)
        else:
            # 对目标求和列作自适应裁剪
            if clip_method == "April":
                # 方法一
                df = clip_target_column(df)
            elif clip_method == "SVTP":
                # 方法二
                m_domain = np.int(df[target_column].max()+1)
                max_range = clip_target_column_SVT(df, m_domain*30, m_domain, value_dimen_process, epsilon, colum_list[0], colum_list[1], colum_list[2], colum_list[3])
                df = auto_trip.data_trip(df, target_column, 0, max_range)
                print("处理后的裁剪范围:(" + str(0) + "," + str(max_range) + ")")

            print("裁剪数据时间：" + str(time.perf_counter() - start) + "s")
            start = time.perf_counter()

            # 数据分布
            # for i in range(10):
            #     distfit_judge(df[i])

            print("通过二维边际分布估计四维合成数据集")
            # 预测分布
            best_0 = distfit_judge(df[colum_list[0]])
            best_1 = distfit_judge(df[colum_list[1]])
            best_2 = distfit_judge(df[colum_list[2]])
            best_3 = distfit_judge(df[colum_list[3]])
            dict_best = {colum_list[0]:best_0,
                        colum_list[1]: best_1,
                        colum_list[2]: best_2,
                        colum_list[3]: best_3,}

            AB_origin, AB_noise, AB_estimate = query_noise_with_estimate(df, colum_list[0], colum_list[1],clip_method)
            AC_origin, AC_noise, AC_estimate = query_noise_with_estimate(df, colum_list[0], colum_list[2],clip_method)
            AD_origin, AD_noise, AD_estimate = query_noise_with_estimate(df, colum_list[0], colum_list[3],clip_method)
            BC_origin, BC_noise, BC_estimate = query_noise_with_estimate(df, colum_list[1], colum_list[2],clip_method)
            BD_origin, BD_noise, BD_estimate = query_noise_with_estimate(df, colum_list[1], colum_list[3],clip_method)
            CD_origin, CD_noise, CD_estimate = query_noise_with_estimate(df, colum_list[2], colum_list[3],clip_method)
            # estimate是估计得到的四维数据
            estimate = max_entropy.max_entropy_2_to_4(AB_estimate, AC_estimate, AD_estimate, BC_estimate, BD_estimate, CD_estimate,np.sum(D_origin), [value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process])
            # estimate = max_entropy.max_entropy_2_to_4(AB_origin, AC_origin, AD_origin, BC_origin, BD_origin, CD_origin,np.sum(D_origin), [value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process])

            print("估计四维合成数据集大小,size" + str(estimate.size))
            print("估计四维合成数据集时间：" + str(time.perf_counter() - start) + "s")
            start = time.perf_counter()

        # 扩展原本10*10*10*10四维数据集到30维
        if not opt.enable_external:
            estimate = count_query.expand_data(estimate, value_dimen_origin, value_dimen_process)
        D_origin = count_query.expand_data(D_origin, value_dimen_origin, value_dimen_process)
        start = time.perf_counter()

        # 计算平均误差
        # err, eval_num = compute_error(estimate, D_origin, p)
        err, eval_num = compute_random_error(estimate, D_origin, [10,100,1000,10000])
        # for q in range(1, 11):
        #     compute_error(estimate, D_origin, p, q / 10)

        # 保存矩阵
        start = time.perf_counter()
        cube_filename = "cube_"+str(colum_list)+"_d"+str(value_dimen_origin)+"_e1_"+datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H%M%S')
        estimate.dump("./data/cube/"+cube_filename+".dat")
        # D_origin.dump("real.dat")
        print("合成数据集存储时间：" + str(time.perf_counter() - start) + "s")
        start = time.perf_counter()

        # 保存矩阵为csv
        count_query.save_csv(estimate,cube_filename,value_dimen_origin)

        # # 输出实验结果
        # with open("IPUMS-result.txt", 'a', encoding='utf-8') as f_six:
        #     f_six.write("epsilon={},dom(D)={},target(M)={},n={},clip_method={},result={}\n"
        #                 .format(epsilon, value_dimen_origin, target_column, n, clip_method, err.sum()/eval_num))
