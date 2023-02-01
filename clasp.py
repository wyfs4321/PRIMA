import pandas as pd
import pprint
from scipy import optimize


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
    df.columns = ['age', 'work', 'no', 'edu', 'edu_time', 'wedding', 'job', 'home', 'race', 'sex', 'income',
                  'loss', 'worktime', 'nation', 'salary']
    pprint.pprint(df.head(5))
    return df


def get_avg_std_from_data(data, index):
    # print(data[index])

    if data[index].dtype != 'object':
        m = data[index].mean()
        t = data[index].std()
        median = data[index].median()
    return int(m), int(t)


def get_min_max_from_data(data, index):

    if data[index].dtype != 'object':
        tmin = data[index].min()
        tmax = data[index].max()
        # print('m:'+str(m)+',t:'+str(t))
    return tmin, tmax


def data_trip(data, index, tmin, tmax):
    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > tmax, index] = tmax
    truncated_data.loc[truncated_data[index] < tmin, index] = tmin

    return truncated_data



def compute_tao_spend(x, args):
    tao = x[0]
    data, index, m, epsilon = args

    truncated_data = data.copy()
    sum_max = (truncated_data.loc[truncated_data[index] > int(m + tao), index] - int(m + tao)).sum()
    sum_min = (truncated_data.loc[truncated_data[index] < int(m - tao), index] - int(m - tao)).sum()
    # print(sum_max, sum_min)
    result = abs(sum_max)+abs(sum_min)+ 2 * pow((m+tao)/epsilon,2)
    # print(tao, result)
    return result


def compute_range_sum(data, index, min_range, max_range):
    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > max_range, index] = max_range
    truncated_data.loc[truncated_data[index] < min_range, index] = min_range
    # print(sum_max, sum_min)
    # print(truncated_data.loc[:, index].sum())
    return truncated_data.loc[:, index].sum()



def auto_clip(data, index, m, tmax, epsilon):
    x = [1]
    res = optimize.minimize(compute_tao_spend, x, method='COBYLA', args=[data, index, m, epsilon],
                            bounds=[(1, tmax-m)])

    return int(res.x)

#ssvt
def no_bias(data, index, min_range, max_range):
    bs = range(int(min_range), int(max_range))
    threshold = 10e9
    best_bs = min_range

    standard_sum = data.loc[:, index].sum()


    for b in bs:
        r = compute_range_sum(data, index, b, max_range)
        # print("test_min_range="+str(b)+",result="+str(r))

        # if the new answer is pretty close to the old answer, stop
        if abs(r - standard_sum) <= threshold:
            threshold = abs(r-standard_sum)
            best_bs = b

    return best_bs



if __name__ == "__main__":

    data = read_data_1()
    index = 'worktime'
    tmin, tmax = get_min_max_from_data(data, index)
    print(tmin, tmax)
    m, t = get_avg_std_from_data(data, index)


    tao = auto_clip(data, index, m, tmax)


    if m-tao<=tmin and m+tao<tmax:
        best_min_range = no_bias(data, index, tmin, m+tao)
    elif m-tao>=tmin and m+tao==tmax:




