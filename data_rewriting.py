import numpy as np
import pandas as pd
import pprint
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "./data/"
epsilon = 0.5
max_times_noise = 1000


def read_data_1():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(data_dir + 'adult/adult.data', header=None, encoding='utf-8')
    df.columns = ['age', 'work', 'no', 'edu', 'edu_time', 'wedding', 'job', 'home', 'race', 'sex', 'income',
                  'loss', 'worktime', 'nation', 'salary']
    pprint.pprint(df.head(5))
    return df


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
    return df


def get_avg_std_from_data(data, index):
    # print(data[index])
    if data[index].dtype == 'int64':
        m = data[index].mean()
        t = data[index].std()
        median = data[index].median()
        print('m:' + str(m) + ',t:' + str(t) + ',median:' + str(median))
    return m, t


def get_min_max_from_data(data, index):
    # print(data[index])
    if data[index].dtype == 'int64':
        min = data[index].min()
        max = data[index].max()
        # print('m:'+str(m)+',t:'+str(t))
    return min, max


def query_sum_double_truncated(data, m, t, index, filter, value, data_num):

    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > int(m + t), index] = int(m + t)
    truncated_data.loc[truncated_data[index] < int(m - t), index] = int(m - t)

    if data_num == 0:
        sum_real = data.loc[data[filter] == value, index].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].sum()
    else:
        sum_real = data.loc[data[filter] == value, index].loc[0:data_num-1].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].loc[0:data_num-1].sum()

    return sum_real, sum_trunc


def query_sum_single_truncated(data, t, index, filter, value,data_num):

    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > int(t), index] = int(t)

    if data_num==0:
        sum_real = data.loc[data[filter] == value, index].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].sum()
    else:
        sum_real = data.loc[data[filter]==value,index].loc[0:data_num-1].sum()
        sum_trunc = truncated_data.loc[truncated_data[filter]==value,index].loc[0:data_num-1].sum()
    return sum_real, sum_trunc


if __name__ == "__main__":

    data = read_data_2()

    index = 'FAMINC'
    filter = 'HEALTH'
    value = 1
    min, max = get_min_max_from_data(data, index)
    print(min, max)
    m, t = get_avg_std_from_data(data, index)


    x_label=[]
    y_no_trunc=[]
    y_double_trunc=[]
    y_single_trunc=[]
    y_result=[]
    y_label=[]

    data_num =0

    err_relevant = []

    sum_real, sum_trunc_double = query_sum_double_truncated(data, m, t, index, filter, value,data_num)
    sum_noise_double = sum_trunc_double + np.random.laplace(0, (m + t) / epsilon)
    err_relevant.append((sum_noise_double - sum_real) / sum_real)
    y_double_trunc.append(abs(err_relevant[-1]))

    x_label.append(data_num)
    y_result.append(abs(err_relevant[-1]))
    y_label.append('double-sided clipping')


    sum_noise_no_trunc = sum_real + np.random.laplace(0, max / epsilon)
    err_relevant.append((sum_noise_no_trunc - sum_real) / sum_real)
    y_no_trunc.append(abs(err_relevant[-1]))
    x_label.append(data_num)
    y_result.append(abs(err_relevant[-1]))
    y_label.append('no clipping')

    sum_real, sum_trunc_single = query_sum_single_truncated(data, m+t, index, filter, value,data_num)
    sum_noise_single = sum_trunc_single + np.random.laplace(0, (m+t) / epsilon)
    err_relevant.append((sum_noise_single - sum_real) / sum_real)
    y_single_trunc.append(abs(err_relevant[-1]))

    x_label.append(data_num)
    y_result.append(abs(err_relevant[-1]))
    y_label.append('single-sided clipping')

    result_df=pd.DataFrame({'x':x_label,'y':y_result,'type':y_label})
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False  
    sns.set(font='SimHei', font_scale=1.5)  
    sns.lineplot(x='x',y='y',hue='type',data=result_df)



    plt.show()

