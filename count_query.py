import max_entropy
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
# 数据集的原始值域
value_dimen_origin = 30
# 数据集压缩后的值域
value_dimen_processed = 10
# 处理前的数据文件相对路径
source_filename = 'virtual/data_adult_30-50-1M.csv'
# 处理完成的数据文件相对路径
process_filename = 'processed/data_adult_30to10-50-1M.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--enable_external', action='store_true', help='读取已经保存的四维数据')
parser.add_argument('--data_path', default='./virtual.dat', help='四维数据路径')
parser.add_argument('--column_1', default=0, help='四维数据列1')
parser.add_argument('--column_2', default=1, help='四维数据列2')
parser.add_argument('--column_3', default=2, help='四维数据列3')
parser.add_argument('--column_4', default=3, help='四维数据列4')

opt, unknown = parser.parse_known_args()
colum_list = [opt.column_1, opt.column_2, opt.column_3, opt.column_4]

# 功能：读取数据集
# 输出：df——返回读取的数据集的dataFrame形式
def read_data_virtual():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(data_dir + process_filename, header=None, encoding='utf-8')
    pprint.pprint(df.head(5))
    print(df.__len__())
    return df

# 功能：从df数据集中查询属性为a的列，返回实际一维边际分布D_origin和加噪后的一维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a——需要统计的属性
# 输出：D_origin——查询得到的实际一维边际分布
#       D_noise——加噪后的一维边际分布
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


# 功能：从df数据集中查询属性为a和属性为b的列，返回实际二维边际分布D_origin和加噪后的二维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a,b——需要统计的属性名称
# 输出：D_origin——查询得到的实际二维边际分布
#       D_noise——加噪后的二维边际分布
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


# 功能：从df数据集中查询属性为a、b、c、d的列，返回实际四维边际分布D_origin和加噪后的四维边际分布D_noise
# 输入：df——需要进行遍历的数据集
#       a,b,c,d——需要统计的属性名称
# 输出：D_origin——查询得到的实际四维边际分布
#       D_noise——加噪后的四维边际分布
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

# 功能：计算四维合成数据集A与四维实际分布B的相对误差
# 输入：A——通过估计得到的四维合成数据集
#       B——查询得到的四维实际分布
# 输出：matrix——返回的相对误差矩阵，可通过mean得到总体平均相对误差
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
    print("相对误差")
    print(matrix)
    # print(matrix.sum()/sum_err)
    return matrix

# 功能：计算四维合成数据集A与四维实际分布B的相对误差
# 输入：A——通过估计得到的四维合成数据集
#       B——查询得到的四维实际分布
#       q——聚合的精度概率
#       p——误差计算公式的参数
#       value_dimen_origin——原始数据维度
#       n ——数据量
# 输出：result_matrix——返回的相对误差矩阵，可通过mean得到总体平均相对误差
def compute_q_error(A, B, q_, p, value_dimen_origin, n, method=0):
    q = int(value_dimen_origin * q_)
    print(q)
    # 求出最终维度
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

# 功能：计算四维合成数据集A与四维实际分布B的绝对误差
# 输入：A——通过估计得到的四维合成数据集
#       B——查询得到的四维实际分布
#       q——聚合的精度概率
#       p——误差计算公式的参数
#       value_dimen_origin——原始数据维度
#       n ——数据量
# 输出：result_matrix——返回的绝对误差矩阵，可通过mean得到总体平均绝对误差
def compute_MNAE(A, B, q_, p, value_dimen_origin, n, method=0):
    q = int(value_dimen_origin * q_)
    print(q)
    # 求出最终维度
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


# 功能：将合成的四维数据集扩展到目标维度
# 输入：dataset——四维合成数据集
#       target_dimen——目标维度
# 输出：result——返回的扩展后的四维合成数据集
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

# 功能：数据值域压缩预处理
# 无输入输出，只是函数的封装
def data_preprocess_fun():
    print("数据预处理")
    data_high = data_preprocess.read_data_high_level(data_dir + source_filename)
    data_low = data_preprocess.process_high_to_low(data_high, value_dimen_origin, value_dimen_processed)
    data_low.to_csv(data_dir + process_filename, header=0, index=0)
    return

# 功能：将输入矩阵保存为csv
# 输入：matrix——需要保存的矩阵
#       cube_filename——生成的文件名称前缀
#       value_dimen_origin——矩阵的大小
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

    # 数据预处理
    data_preprocess_fun()
    start = time.clock()

    # 读取数据
    df = read_data_virtual()
    print("读取数据时间："+str(time.clock()-start)+"s")
    start = time.clock()
    if opt.enable_external:
        # 通过磁盘存储的已合成数据集读取估计四维合成数据集
        estimate = np.load(opt.data_path, allow_pickle=True)
    else:
        print("通过二维边际分布估计四维合成数据集")
        AB_origin, AB_noise = query_generate_dimension_two(df, colum_list[0],colum_list[1])
        AC_origin, AC_noise = query_generate_dimension_two(df, colum_list[0],colum_list[2])
        AD_origin, AD_noise = query_generate_dimension_two(df, colum_list[0],colum_list[3])
        BC_origin, BC_noise = query_generate_dimension_two(df, colum_list[1],colum_list[2])
        BD_origin, BD_noise = query_generate_dimension_two(df, colum_list[1],colum_list[3])
        CD_origin, CD_noise = query_generate_dimension_two(df, colum_list[2],colum_list[3])

        # estimate是计数值
        estimate = max_entropy.max_entropy_2_to_4(AB_noise, AC_noise, AD_noise, BC_noise, BD_noise, CD_noise, n,
                                                  [value_dimen_processed, value_dimen_processed, value_dimen_processed, value_dimen_processed])
        print("估计四维合成数据集大小,size:" + str(estimate.size))
        print(estimate)
        print("估计四维合成数据集时间："+str(time.clock()-start)+"s")

    # 四维数组
    start = time.clock()
    D_origin, D_noise = query_generate_dimension_four(df, colum_list[0],colum_list[1],colum_list[2],colum_list[3])
    print("查询四维实际分布大小,size:" + str(D_origin.size))
    print(D_origin)
    print("查询四维实际分布时间："+str(time.clock()-start)+"s")

    # 扩展原本10*10*10*10四维数据集到30维
    if not opt.enable_external:
        estimate = expand_data(estimate, value_dimen_origin, value_dimen_processed)
    D_origin = expand_data(D_origin, value_dimen_origin, value_dimen_processed)

    # 计算查询范围为q的误差均值
    # A = np.ones((10,10,10,10))
    # B = np.zeros((10,10,10,10))
    start = time.clock()
    print("q=10%误差")
    err_10,_ = compute_q_error(estimate, D_origin, 0.1, p, value_dimen_origin, n)
    # 查询次数
    query_time = value_dimen_origin - value_dimen_origin * 0.1 + 1
    print("平均误差："+str(err_10.mean()))
    print("目前耗时："+str(time.clock()-start)+"s")
    print("查询次数："+str(pow(query_time, 4)))

    start = time.clock()
    print("q=20%误差")
    err_20,_ = compute_q_error(estimate, D_origin, 0.2, p, value_dimen_origin, n)
    print("平均误差：" + str(err_20.mean()))
    # 查询次数
    query_time = value_dimen_origin - value_dimen_origin * 0.2 + 1
    print("目前耗时：" + str(time.clock() - start) + "s")
    print("查询次数：" + str(pow(query_time, 4)))

    start = time.clock()
    print("q=30%误差")
    err_25,_ = compute_q_error(estimate, D_origin, 0.3, p, value_dimen_origin, n)
    print("平均误差：" + str(err_25.mean()))
    # 查询次数
    query_time = value_dimen_origin - value_dimen_origin * 0.3 + 1
    print("目前耗时：" + str(time.clock() - start) + "s")
    print("查询次数：" + str(pow(query_time, 4)))

    start = time.clock()
    print("q=50%误差")
    err_50,_ = compute_q_error(estimate, D_origin, 0.5, p, value_dimen_origin, n)
    print("平均误差：" + str(err_50.mean()))
    # 查询次数
    query_time = value_dimen_origin - value_dimen_origin * 0.5 + 1
    print("目前耗时：" + str(time.clock() - start) + "s")
    start = time.clock()
    print("查询次数：" + str(pow(query_time, 4)))

    start = time.clock()
    print("q=70%误差")
    err_70,_ = compute_q_error(estimate, D_origin, 0.7, p, value_dimen_origin, n)
    print("平均误差：" + str(err_70.mean()))
    # 查询次数
    query_time = value_dimen_origin - value_dimen_origin * 0.7 + 1
    print("目前耗时：" + str(time.clock() - start) + "s")
    print("查询次数：" + str(pow(query_time, 4)))

    start = time.clock()
    print("q=100%误差")
    err_100,_ = compute_q_error(estimate, D_origin, 1, p, value_dimen_origin, n)
    print("平均误差：" + str(err_100.mean()))
    print("目前耗时：" + str(time.clock() - start) + "s")
    print("查询次数：" + str(1))

    ## 保存矩阵
    start = time.clock()
    cube_filename = "cube_"+str(colum_list)+"_d"+str(value_dimen_origin)+"_e1_"+datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H%M%S')
    estimate.dump("./data/cube/"+cube_filename+".dat")
    # D_origin.dump("real.dat")
    print("合成数据集存储时间：" + str(time.clock() - start) + "s")
    start = time.clock()

    # 保存矩阵为csv
    save_csv(estimate,cube_filename,value_dimen_origin)

    # 模拟查询时间
    # test_Data = np.load("virtual.dat", allow_pickle=True)
    # for q in range(10):
    #     print(test_Data[0:0 + q, 0:0 + q, 0:0 + q, 0:0 + q].sum())
    # print("十次模拟查询时间：" + str(time.clock() - start) + "s")
    # start = time.clock()




