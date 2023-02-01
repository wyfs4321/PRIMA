import numpy as np
import pandas as pd
import pprint
import math
from scipy.stats import norm,shapiro,kstest,normaltest,anderson
import scipy.stats as stats
import scipy
import sum_query_processing
import clasp
import argparse
import cuboid_constructing
import data_preprocessing
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
# parameter
alpha = 1.0
beta = 0.0
max_f = 0.01
scale = 1

def str2bool(str):
    return True if str.lower()=='true' else False

parser = argparse.ArgumentParser()
parser.add_argument('--enable_external', action='store_true')
parser.add_argument('--data_path', default='./virtual.dat')
parser.add_argument('--column_1', default=0)
parser.add_argument('--column_2', default=1)
parser.add_argument('--column_3', default=2)
parser.add_argument('--column_4', default=3)
parser.add_argument('--target_column',default=12,type=int)
parser.add_argument('--alpha',default=1, type=float)
parser.add_argument('--beta',default=0, type=float)
parser.add_argument('--maxf',default=0.01, type=float)
parser.add_argument('--n',default=100000, type=int)
parser.add_argument('--disable_maxentropy', type=str2bool,default="False")
parser.add_argument('--disable_prefixSum', action='store_true')
parser.add_argument('--clip_method', default='April')
parser.add_argument('--epsilon', default=3,type=float)
parser.add_argument('--value_dimen_origin', default=30,type=int)
parser.add_argument('--dataset', default='IPUMS')



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


value_dimen_origin = opt.value_dimen_origin
value_dimen_process = 10

source_filename = 'virtual/'+ opt.dataset +'_10-'+ str(value_dimen_origin)+ '.csv'

process_filename = 'processed/'+ opt.dataset+ '_10-'+ str(value_dimen_origin)+ 'to10.csv'


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


def data_preprocess_fun():
    data_high = data_preprocess.read_data_high_level(data_dir + source_filename)
    data_low = data_preprocess.process_high_to_low(data_high, value_dimen_origin, value_dimen_process)
    data_low.to_csv(data_dir + process_filename, header=0, index=0)
    return



def query_generate_dimension_two(df, a, b):
    D_origin = np.zeros((value_dimen_process, value_dimen_process))
    for i in range(value_dimen_process):
        for j in range(value_dimen_process):
            D_origin[i][j] = df.loc[(df[a] == i) & (df[b] == j), target_column].sum()
    f = df[target_column].max()
    D_noise = D_origin + np.random.laplace(0, f / epsilon, (value_dimen_process, value_dimen_process))
    # print(D_origin)
    # print(D_noise)
    return D_origin,D_noise


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

    f = df[target_column].max()
    D_noise = D_origin + np.random.laplace(0, f / epsilon, (value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process))
    return D_origin, D_noise


def clip_target_column(df):
    tmin, tmax = auto_trip.get_min_max_from_data(df, target_column)
    m, t = auto_trip.get_avg_std_from_data(df, target_column)

    tao = auto_trip.auto_clip(df, target_column, m, tmax, epsilon)


    if m - tao <= tmin and m + tao < tmax:
        best_min_range = auto_trip.no_bias(df, target_column, tmin, m + tao)
        df = auto_trip.data_trip(df, target_column, best_min_range, m + tao)
    elif m + tao >= tmax:
        print("cliping range:(" + str(tmin) + "," + str(tmax) + ")")
    else:
        df = auto_trip.data_trip(df, target_column, m - tao, m + tao)
        print("cliping range:(" + str(m - tao) + "," + str(m + tao) + ")")
    return df


def difference_query(df,domain,Da,Db,b,epsilon_i):
    summation=0
    for i in range(domain):
        for j in range(domain):
            summation+=df.loc[(df[Da] == i) & (df[Db] == j), target_column].sum() + np.random.laplace(0, b / epsilon_i)
            # summation+=df.loc[(df[Da] == i) & (df[Db] == j), target_column].sum()

    return summation




def fitter_judge(table):
    f = Fitter(table,distributions = ['uniform','norm']) 
    f.fit()  
    result = f.summary(10) 
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


def f(x):
    return 1.0 / pow(2*math.pi,0.5) * pow(math.e,-(x*x/2))


def f2(i,j, u1, u2, sigma1, sigma2):
    x = scipy.stats.multivariate_normal(mean=np.array([0, 0]),cov=np.array([[1,0],[0,1]])).pdf(
        x=np.array([(j - u2 - beta * sigma2) / sigma2 * (1/u2), (i - u1 - beta * sigma1) / sigma1 * (1/u1)]))
    # return max(max_f, x)
    return x


def f1(i,j, u1, u2, sigma1, sigma2):
    return 1


def F2(i, j, u1, u2, sigma1, sigma2):
    x = scipy.stats.multivariate_normal(mean=np.array([0,0]),cov=np.array([[1,0],[0,1]])).cdf(
        x=np.array([(j - u2 - beta * sigma2) / sigma2 * (1/u2), (i - u1 - beta * sigma1) / sigma1 * (1/u1)]))

    return x


def F1(i, j, u1, u2, sigma1, sigma2):
    return (i+1)*(j+1)


def P(x):
    return norm.cdf(x) 


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


def getU(matrix):
    mean = 0.0
    for i in range(matrix.__len__()):
        mean = mean + matrix[i]*i
    return mean/np.sum(matrix)


def getSigma(matrix, U):
    sigma = 0.0
    for i in range(matrix.__len__()):
        sigma = sigma + math.pow((U-i),2)* matrix[i]
    return math.sqrt(sigma/np.sum(matrix))


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


def compute_mij(matrix):
    new_matrix = np.zeros((matrix.__len__(),matrix.__len__()))
    for i in range(matrix.__len__()):
        for j in range(matrix.__len__()):
            for k in range(i,j+1):
                new_matrix[i,j]=new_matrix[i,j]+matrix[k]/(j-i+1)
    # print(new_matrix)
    return new_matrix


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


def query_noise_with_estimate(data, A, B, clip_method=None):

    origin, noise = query_generate_dimension_two(data, A, B)

    if opt.disable_prefixSum:
        return origin, noise, noise

    prefix_sum = get_prefix_sum(origin)
    prefix_origin = get_original_matrix(prefix_sum)

    noise = np.maximum(noise, 0)

    estimate = estimate_sum_two_dimens(noise, dict_best[A], dict_best[B])
    estimate = post_process_double(estimate)
    # estimate = normalize(estimate, prefix_sum)
    estimate = get_original_matrix(estimate)
    # estimate = normalize(estimate, origin)

    return origin, noise, estimate


def compute_error(A, B, p, q=None):
    start = time.perf_counter()
    if q is None:
        err, eval_num = count_query.compute_MNAE(A, B, 1/value_dimen_origin, p, value_dimen_origin, B.sum(), method=1)
    else:
        err, eval_num = count_query.compute_MNAE(A, B, q, p, value_dimen_origin, B.sum(),method=1)


    query_time = value_dimen_origin
    return err, eval_num


def compute_random_error(A, B, n_list):
    start = time.perf_counter()

    err = A-B
    err_use = np.array(err).flatten()

    for union in n_list:
        select = np.random.choice(err_use, size=union, replace=False)
        with open("clip-result.txt", 'a', encoding='utf-8') as f_six:
            f_six.write("dataset={},epsilon={},dom(D)={},target(M)={},n={},clip_method={},union={},result={}\n"
                        .format(opt.dataset, epsilon, value_dimen_origin, target_column, n, clip_method, union, select.mean()))
    return select, n

if __name__ == "__main__":
    start = time.perf_counter()

    data_preprocess_fun()
    start = time.perf_counter()

    df = read_data_virtual()
    start = time.perf_counter()

    D_origin, D_noise = query_generate_dimension_four(df, colum_list[0], colum_list[1], colum_list[2], colum_list[3])

    start = time.perf_counter()

    if opt.disable_maxentropy:

        if clip_method == "April":
            df = clip_target_column(df)
            clip_origin, clip_noise = query_generate_dimension_four(df, colum_list[0], colum_list[1], colum_list[2], colum_list[3])

        start = time.perf_counter()
        D_origin = count_query.expand_data(D_origin, value_dimen_origin, value_dimen_process)
        D_noise = count_query.expand_data(D_noise, value_dimen_origin, value_dimen_process)
        
        # err, eval_num = compute_error(D_noise, D_origin, p)
        err, eval_num = compute_random_error(clip_noise, D_origin, [10,100,1000,10000])
        # for q in range(1,11):
        #     compute_error(D_noise,D_origin, p, q/10)


    else:
        if opt.enable_external:
            estimate = np.load(opt.data_path, allow_pickle=True)
        else:
            if clip_method == "April":
                df = clip_target_column(df)


            start = time.perf_counter()

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
            estimate = max_entropy.max_entropy_2_to_4(AB_estimate, AC_estimate, AD_estimate, BC_estimate, BD_estimate, CD_estimate,np.sum(D_origin), [value_dimen_process, value_dimen_process, value_dimen_process, value_dimen_process])

            start = time.perf_counter()

        if not opt.enable_external:
            estimate = count_query.expand_data(estimate, value_dimen_origin, value_dimen_process)
        D_origin = count_query.expand_data(D_origin, value_dimen_origin, value_dimen_process)
        start = time.perf_counter()

        # err, eval_num = compute_error(estimate, D_origin, p)
        err, eval_num = compute_random_error(estimate, D_origin, [10,100,1000,10000])
        # for q in range(1, 11):
        #     compute_error(estimate, D_origin, p, q / 10)

        start = time.perf_counter()
        cube_filename = "cube_"+str(colum_list)+"_d"+str(value_dimen_origin)+"_e1_"+datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H%M%S')
        estimate.dump("./data/cube/"+cube_filename+".dat")
        # D_origin.dump("real.dat")
        start = time.perf_counter()

        count_query.save_csv(estimate,cube_filename,value_dimen_origin)

