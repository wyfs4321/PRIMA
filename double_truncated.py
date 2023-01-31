import numpy as np
import pandas as pd
import pprint
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "./data/"
epsilon = 0.5
max_times_noise = 1000

# 读取adult数据
def read_data_1():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(data_dir + 'adult/adult.data', header=None, encoding='utf-8')
    df.columns = ['年龄', '工作类型', '序号', '教育程度', '受教育时间', '婚姻状况', '职业', '家庭关系', '人种', '性别', '资本收益',
                  '资本损失', '每周工作时间', '祖国', '收入']
    pprint.pprint(df.head(5))
    return df

# 读取dataset2数据
def read_data_2():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(data_dir + 'data2/dataset_2.csv', encoding='utf-8')
    df.columns = ['YEAR', 'MONTH', 'AGE', 'EMPSTAT', 'AHRSWORKT', 'HEALTH', 'FAMINC']
    pprint.pprint(df.head(5))
    pprint.pprint(df['FAMINC'].value_counts())
    #sns.distplot(df['FAMINC'], hist=True, bins=10000, kde=False)
    #df['FAMINC'].value_counts().plot.bar()
    #plt.show()
    # 可用：age/AHRSWORKT/FAMINC，分类：HEALTH
    return df

# 获得数据中位数标准差
def get_avg_std_from_data(data, index):
    # print(data[index])
    # 属于连续数据
    if data[index].dtype == 'int64':
        m = data[index].mean()
        t = data[index].std()
        median = data[index].median()
        print('m:' + str(m) + ',t:' + str(t) + ',median:' + str(median))
    return m, t

# 获得数据最小值最大值
def get_min_max_from_data(data, index):
    # print(data[index])
    # 属于连续数据
    if data[index].dtype == 'int64':
        min = data[index].min()
        max = data[index].max()
        # print('m:'+str(m)+',t:'+str(t))
    return min, max

# 双边截断，根据data_num规模或者filter进行过滤
def query_sum_double_truncated(data, m, t, index, filter, value, data_num):
    # 数据进行双边截断
    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > int(m + t), index] = int(m + t)
    truncated_data.loc[truncated_data[index] < int(m - t), index] = int(m - t)
    # 过滤求和
    if data_num == 0:
        sum_real = data.loc[data[filter] == value, index].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].sum()
    else:
        sum_real = data.loc[data[filter] == value, index].loc[0:data_num-1].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].loc[0:data_num-1].sum()
    # sum_real = data.loc[data[filter], index].sum()
    # sum_trunc = truncated_data.loc[truncated_data[filter], index].sum()
    # print('sum_trunc:'+str(sum_trunc)+',sum_real:'+str(sum_real))
    # 加噪
    return sum_real, sum_trunc

# 单边截断，根据data_num规模或者filter进行过滤
def query_sum_single_truncated(data, t, index, filter, value,data_num):
    # 数据进行单边截断
    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > int(t), index] = int(t)
    # 过滤求和
    if data_num==0:
        sum_real = data.loc[data[filter] == value, index].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].sum()
    else:
        sum_real = data.loc[data[filter]==value,index].loc[0:data_num-1].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter]==value,index].loc[0:data_num-1].sum()
    # print('sum_trunc:'+str(sum_trunc)+',sum_real:'+str(sum_real))
    # 加噪
    return sum_real, sum_trunc


if __name__ == "__main__":
    # 读取数据
    data = read_data_2()
    # # 获取属性i的m\t值
    # for index in data.columns:
    #     if (data[index].dtype=='int64'):
    #         m,t = get_avg_std_from_data(data,index)
    #         for filter in data.columns:
    #             if (data[filter].dtype=='object'):
    #                 for value in (data[filter].unique()):
    #                     # 根据m\t值进行双边截断的求和查询,其中筛选条件是j列的第k项
    #                     sum_real, sum_trunc = query_sum_double_truncated(data,m,t,index,filter,value)
    #                     # 加多次噪声取相对误差均值
    #                     err_relevant=[]
    #                     for time in range(max_times_noise):
    #                         sum_noise = sum_trunc + np.random.laplace(0, (m + t) / epsilon)
    #                         err_relevant.append((sum_noise-sum_real)/sum_real)
    #                     # print("求和:"+str(index)+",筛选条件:"+str(filter)+","+str(value)
    #                     #       +"，相对误差:"+str(np.mean(err_relevant)))
    #                     print(str(index) + "，" + str(filter) + "，" + str(value)
    #                           + "，" + str(np.mean(err_relevant)))
    index = 'FAMINC'
    filter = 'HEALTH'
    value = 1
    min, max = get_min_max_from_data(data, index)
    print(min, max)
    m, t = get_avg_std_from_data(data, index)

    # 画图准备
    x_label=[]
    y_no_trunc=[]
    y_double_trunc=[]
    y_single_trunc=[]
    y_result=[]
    y_label=[]

    data_num =0
    # for data_num in range(10,1000,10):
    #     print("*************数据规模:"+str(data_num)+"************")
    err_relevant = []
    # 双边截断实验结果
    sum_real, sum_trunc_double = query_sum_double_truncated(data, m, t, index, filter, value,data_num)
    sum_noise_double = sum_trunc_double + np.random.laplace(0, (m + t) / epsilon)
    err_relevant.append((sum_noise_double - sum_real) / sum_real)
    y_double_trunc.append(abs(err_relevant[-1]))

    x_label.append(data_num)
    y_result.append(abs(err_relevant[-1]))
    y_label.append('双边截断')

    # 不截断实验结果
    sum_noise_no_trunc = sum_real + np.random.laplace(0, max / epsilon)
    err_relevant.append((sum_noise_no_trunc - sum_real) / sum_real)
    y_no_trunc.append(abs(err_relevant[-1]))
    x_label.append(data_num)
    y_result.append(abs(err_relevant[-1]))
    y_label.append('不截断')
    # 单边截断实验结果
    sum_real, sum_trunc_single = query_sum_single_truncated(data, m+t, index, filter, value,data_num)
    sum_noise_single = sum_trunc_single + np.random.laplace(0, (m+t) / epsilon)
    err_relevant.append((sum_noise_single - sum_real) / sum_real)
    y_single_trunc.append(abs(err_relevant[-1]))

    x_label.append(data_num)
    y_result.append(abs(err_relevant[-1]))
    y_label.append('单边截断')

        # print('双边截断:'+str(err_relevant[0]))
        # print('不截断:' + str(err_relevant[1]))
        # print('############真实结果############')
        # print('求和结果:' + str(sum_real))
        # print('加噪结果:' + str(sum_noise_no_trunc))
        # print('误差:' + str(err_relevant[-1]))
        # print('############双边截断结果############')
        # print('求和结果:' + str(sum_trunc_double))
        # print('加噪结果:' + str(sum_noise_double))
        # print('误差:' + str(err_relevant[-2]))
        # print("")
        # print('双边截断求和结果:' + str(sum_trunc_double))

    #画图
    result_df=pd.DataFrame({'x':x_label,'y':y_result,'type':y_label})
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    sns.set(font='SimHei', font_scale=1.5)  # 解决Seaborn中文显示问题并调整字体大小
    sns.lineplot(x='x',y='y',hue='type',data=result_df)



    plt.show()
    # 单边截断实验结果
    # 对于不同t值
    # for i in range(min,max+1,49):
    #     sum_real, sum_trunc_single = query_sum_single_truncated(data, i, index, filter, value)
    #     sum_noise_single = sum_trunc_single + np.random.laplace(0, i / epsilon)
    #     err_relevant.append((sum_noise_single - sum_real) / sum_real)
    #     print('############单边截断结果############')
    #     print('t='+str(i)+"求和结果:" + str(sum_trunc_single))
    #     print("\t\t加噪结果:" + str(sum_noise_single))
    #     print('\t\t误差:' + str(err_relevant[-1]))
