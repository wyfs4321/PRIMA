import construction
import pandas as pd
import numpy as np
import pprint
import time
import argparse
import datetime
import csv
import data_preprocess

data_dir = "./data/"
epsilon = 1
max_times_noise = 1000
n = 1000000
p = 0.01
value_dimen_origin = 30

value_dimen_processed = 10

source_filename = 'virtual/data_adult_30-50-1M.csv'

process_filename = 'processed/data_adult_30to10-50-1M.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--enable_external', action='store_true')
parser.add_argument('--data_path', default='./virtual.dat')
parser.add_argument('--column_1', default=0)
parser.add_argument('--column_2', default=1)
parser.add_argument('--column_3', default=2)
parser.add_argument('--column_4', default=3)

opt, unknown = parser.parse_known_args()
colum_list = [opt.column_1, opt.column_2, opt.column_3, opt.column_4]


def read_data_virtual():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(data_dir + process_filename, header=None, encoding='utf-8')
    pprint.pprint(df.head(5))
    print(df.__len__())
    return df


def query_generate_dimension_one(df, a):
    min_val = df[a].min()
    max_val = df[a].max()
    D = df[a].value_counts()
    # print(D)
    D_origin = np.zeros((1,value_dimen_processed))
    for i in range(value_dimen_processed):
        D_origin[0][i]=D[i]
    # pprint.pprint(D_origin)
    D_noise = D_origin + np.random.laplace(0, 1 / epsilon, (1,value_dimen_processed))
    # print(D_origin)
    # print(D_noise)
    return D_origin,D_noise



def query_generate_dimension_two(df, a, b):
    min_val = df[a].min()
    max_val = df[a].max()
    # D = df[b].groupby(df[a]).value_counts(dropna=False)
    # print(D)
    # D = df[a].groupby(df[b]).value_counts()

    D_origin = np.zeros((value_dimen_processed,value_dimen_processed))
    for i in range(value_dimen_processed):
        for j in range(value_dimen_processed):
            D_origin[i][j] = df.loc[(df[a] == i) & (df[b] == j), :].count()[0]
    D_noise = D_origin + np.random.laplace(0, 1 / epsilon, (value_dimen_processed,value_dimen_processed))
    # print(D_origin)
    print(D_noise)
    return D_origin,D_noise



def query_generate_dimension_four(df, a, b, c, d):
    # min_val = df[a].min()
    # max_val = df[a].max()
    # D = df[c].value_counts(groupby(df[b]).groupby(df[a]).value_counts(dropna=False)
    # D = df[a].groupby(df[b]).value_counts()
    D = df.loc[(df[a]==0) & (df[b]==0) & (df[c]==0) & (df[d]==1), :].count()[0]
    pprint.pprint(D)

    D_origin = np.zeros((value_dimen_processed,value_dimen_processed,value_dimen_processed,value_dimen_processed))
    for i in range(value_dimen_processed):
        for j in range(value_dimen_processed):
            for k in range(value_dimen_processed):
                for l in range(value_dimen_processed):
                    D_origin[i][j][k][l] = df.loc[(df[a]==i) & (df[b]==j) & (df[c]==k) & (df[d]==l), :].count()[0]

    # pprint.pprint(D_origin)
    D_noise = D_origin + np.random.laplace(0, 1 / epsilon, (value_dimen_processed,value_dimen_processed,value_dimen_processed,value_dimen_processed))
    # print(D_origin)
    # print(D_noise)
    return D_origin, D_noise


def compute_relevant_error_4(A, B, p):
    # matrix = A.copy()
    matrix = np.divide(A, np.sum(B)) - np.divide(B, np.sum(B))
    # matrix = A-B
    # sum_err = 0
    for i in range(B.__len__()):
        for j in range(B[i].__len__()):
            for k in range(B[i][j].__len__()):
                for l in range(B[i][j][k].__len__()):
                    if (B[i][j][k][l]==0):
                        matrix[i][j][k][l] = 0
                    else:
                        matrix[i][j][k][l] =abs(matrix[i][j][k][l]/(B[i][j][k][l]/n + p))
                        # sum_err = sum_err+1

    print(matrix)
    # print(matrix.sum()/sum_err)
    return matrix


def compute_q_error(A, B, q_, p, value_dimen_origin, n, method=0):
    q = int(value_dimen_origin * q_)
    print(q)

    # matrix = np.abs(np.divide(A, n) - np.divide(B, n))
    matrix = np.abs(A - B)
    # print(matrix[0,0,0:2,1:2].sum())
    # print(matrix[0:1][0:1][0:1][1:2].sum())
    m = matrix.__len__()
    result_len = m-q+1
    # # 默认四维误差
    result_matrix = np.zeros((result_len, result_len, result_len, result_len))
    eval_err_num = 0
    for i in range(result_len):
        for j in range(result_len):
            for k in range(result_len):
                for l in range(result_len):
                    cur_err = matrix[i:i+q,j:j+q,k:k+q,l:l+q].sum()
                    true_val = B[i:i+q,j:j+q,k:k+q,l:l+q].sum()
                    if method==0:
                        # result_matrix[i][j][k][l] = abs(cur_err/(true_val/n + p))
                        result_matrix[i][j][k][l] = abs(cur_err/(true_val + p))
                        eval_err_num = eval_err_num+1
                    else:
                        if true_val==0:
                            result_matrix[i][j][k][l] = 0
                        else:
                            # result_matrix[i][j][k][l] = abs(cur_err/(true_val/n))
                            result_matrix[i][j][k][l] = abs(cur_err/(true_val))
                            eval_err_num = eval_err_num+1
    # print(result_matrix)
    return result_matrix, eval_err_num


def compute_MNAE(A, B, q_, p, value_dimen_origin, n, method=0):
    q = int(value_dimen_origin * q_)
    print(q)

    matrix = A - B
    # print(matrix[0,0,0:2,1:2].sum())
    # print(matrix[0:1][0:1][0:1][1:2].sum())
    m = matrix.__len__()
    result_len = m - q + 1
    # # 默认四维误差
    result_matrix = np.zeros((result_len, result_len, result_len, result_len))
    eval_err_num = 0
    for i in range(result_len):
        for j in range(result_len):
            for k in range(result_len):
                for l in range(result_len):
                    cur_err = matrix[i:i + q, j:j + q, k:k + q, l:l + q].sum()
                    result_matrix[i][j][k][l] = abs(cur_err / n)
                    eval_err_num = eval_err_num + 1
    # print(result_matrix)
    return result_matrix, eval_err_num



def expand_data(dataset, target_dimen=30, source_dimen=10):
    result = np.zeros((target_dimen, target_dimen, target_dimen, target_dimen))
    for i in range(target_dimen):
        for j in range(target_dimen):
            for k in range(target_dimen):
                for l in range(target_dimen):
                    expand_scale = target_dimen/source_dimen
                    result[i][j][k][l] = dataset[int(i/expand_scale),int(j/expand_scale),int(k/expand_scale),int(l/expand_scale)]/pow(expand_scale, 4)
    # print(result)
    return result


def data_preprocess_fun():

    data_high = data_preprocess.read_data_high_level(data_dir + source_filename)
    data_low = data_preprocess.process_high_to_low(data_high, value_dimen_origin, value_dimen_processed)
    data_low.to_csv(data_dir + process_filename, header=0, index=0)
    return


def save_csv(matrix,cube_filename,value_dimen_origin):
    with open("./data/cube/"+cube_filename+".csv", 'w',newline='') as f:

        writer = csv.writer(f)
        header = ["column1","column2","column3","column4","count"]
        writer.writerow(header)
        for i in range(value_dimen_origin):
            for j in range(value_dimen_origin):
                for k in range(value_dimen_origin):
                    for l in range(value_dimen_origin):
                        data = [str(i),str(j),str(k),str(l),str(matrix[i][j][k][l])]
                        writer.writerow(data)
    return


if __name__ == "__main__":
    start = time.clock()


    data_preprocess_fun()
    start = time.clock()


    df = read_data_virtual()

    start = time.clock()
    if opt.enable_external:

        estimate = np.load(opt.data_path, allow_pickle=True)
    else:

        AB_origin, AB_noise = query_generate_dimension_two(df, colum_list[0],colum_list[1])
        AC_origin, AC_noise = query_generate_dimension_two(df, colum_list[0],colum_list[2])
        AD_origin, AD_noise = query_generate_dimension_two(df, colum_list[0],colum_list[3])
        BC_origin, BC_noise = query_generate_dimension_two(df, colum_list[1],colum_list[2])
        BD_origin, BD_noise = query_generate_dimension_two(df, colum_list[1],colum_list[3])
        CD_origin, CD_noise = query_generate_dimension_two(df, colum_list[2],colum_list[3])


        estimate = max_entropy.max_entropy_2_to_4(AB_noise, AC_noise, AD_noise, BC_noise, BD_noise, CD_noise, n,
                                                  [value_dimen_processed, value_dimen_processed, value_dimen_processed, value_dimen_processed])
        print("size:" + str(estimate.size))
        print(estimate)
        print(str(time.clock()-start)+"s")


    start = time.clock()
    D_origin, D_noise = query_generate_dimension_four(df, colum_list[0],colum_list[1],colum_list[2],colum_list[3])
    print("size:" + str(D_origin.size))
    print(D_origin)
    print(str(time.clock()-start)+"s")


    if not opt.enable_external:
        estimate = expand_data(estimate, value_dimen_origin, value_dimen_processed)
    D_origin = expand_data(D_origin, value_dimen_origin, value_dimen_processed)

    start = time.clock()
    print("q=10%")
    err_10,_ = compute_q_error(estimate, D_origin, 0.1, p, value_dimen_origin, n)

    query_time = value_dimen_origin - value_dimen_origin * 0.1 + 1
    print(str(err_10.mean()))

    start = time.clock()
    print("q=20%")
    err_20,_ = compute_q_error(estimate, D_origin, 0.2, p, value_dimen_origin, n)
    print( str(err_20.mean()))

    query_time = value_dimen_origin - value_dimen_origin * 0.2 + 1

    start = time.clock()
    print("q=30%")
    err_25,_ = compute_q_error(estimate, D_origin, 0.3, p, value_dimen_origin, n)
    print(str(err_25.mean()))

    query_time = value_dimen_origin - value_dimen_origin * 0.3 + 1


    start = time.clock()
    print("q=50%")
    err_50,_ = compute_q_error(estimate, D_origin, 0.5, p, value_dimen_origin, n)
    print(str(err_50.mean()))

    query_time = value_dimen_origin - value_dimen_origin * 0.5 + 1
    start = time.clock()
    start = time.clock()
    print("q=70%")
    err_70,_ = compute_q_error(estimate, D_origin, 0.7, p, value_dimen_origin, n)
    print(str(err_70.mean()))

    query_time = value_dimen_origin - value_dimen_origin * 0.7 + 1


    start = time.clock()
    print("q=100%")
    err_100,_ = compute_q_error(estimate, D_origin, 1, p, value_dimen_origin, n)
    print(str(err_100.mean()))



    start = time.clock()
    cube_filename = "cube_"+str(colum_list)+"_d"+str(value_dimen_origin)+"_e1_"+datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H%M%S')
    estimate.dump("./data/cube/"+cube_filename+".dat")
    # D_origin.dump("real.dat")

    start = time.clock()

    save_csv(estimate,cube_filename,value_dimen_origin)




