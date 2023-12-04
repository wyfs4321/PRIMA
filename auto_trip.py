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
    df.columns = ['年龄', '工作类型', '序号', '教育程度', '受教育时间', '婚姻状况', '职业', '家庭关系', '人种', '性别', '资本收益',
                  '资本损失', '每周工作时间', '祖国', '收入']
    pprint.pprint(df.head(5))
    return df

# 获得数据均值中位数标准差
# 输入：data——需要进行查询的数据集
#       index——需要查询数据信息的属性所在列
# 输出：m——目标列数据的均值
#       t——目标列数据的标准差
def get_avg_std_from_data(data, index):
    # print(data[index])
    # 属于连续数据
    if data[index].dtype != 'object':
        m = data[index].mean()
        t = data[index].std()
        median = data[index].median()
        print('均值:' + str(m) + ',标准差:' + str(t) + ',中位数:' + str(median))
    return int(m), int(t)

# 获得数据最小值最大值
# 输入：data——需要进行查询的数据集
#       index——需要查询数据信息的属性所在列
# 输出：tmin——目标列数据的最小值
#       tmax——目标列数据的最大值
def get_min_max_from_data(data, index):
    # print(data[index].dtype)
    # 属于连续数据
    if data[index].dtype != 'object':
        tmin = data[index].min()
        tmax = data[index].max()
        # print('m:'+str(m)+',t:'+str(t))
    return tmin, tmax

# # 双边截断，根据data_num规模或者filter进行过滤
# def query_sum_double_truncated(data, m, t, index, filter, value, data_num):
#     # 数据进行双边截断
#     truncated_data = data.copy()
#     truncated_data.loc[truncated_data[index] > int(m + t), index] = int(m + t)
#     truncated_data.loc[truncated_data[index] < int(m - t), index] = int(m - t)
#     # 过滤求和
#     if data_num == 0:
#         sum_real = data.loc[data[filter] == value, index].sum()
#         sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].sum()
#     else:
#         sum_real = data.loc[data[filter] == value, index].loc[0:data_num-1].sum()
#         sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].loc[0:data_num-1].sum()
#     # sum_real = data.loc[data[filter], index].sum()
#     # sum_trunc = truncated_data.loc[truncated_data[filter], index].sum()
#     # print('sum_trunc:'+str(sum_trunc)+',sum_real:'+str(sum_real))
#     # 加噪
#     return sum_real, sum_trunc

# 单边截断，根据data_num规模或者filter进行过滤
# def query_sum_single_truncated(data, t, index, filter, value,data_num):
#     # 数据进行单边截断
#     truncated_data = data.copy()
#     truncated_data.loc[truncated_data[index] > int(t), index] = int(t)
#     # 过滤求和
#     if data_num==0:
#         sum_real = data.loc[data[filter] == value, index].sum()
#         sum_trunc = truncated_data.loc[truncated_data[filter] == value, index].sum()
#     else:
#         sum_real = data.loc[data[filter]==value,index].loc[0:data_num-1].sum()
#         sum_trunc = truncated_data.loc[truncated_data[filter]==value,index].loc[0:data_num-1].sum()
#     # print('sum_trunc:'+str(sum_trunc)+',sum_real:'+str(sum_real))
#     # 加噪
#     return sum_real, sum_trunc

# 对数据index列按照(min,max)进行截断
# 输入：data——需要进行查询的数据集
#       index——需要查询数据信息的属性所在列
#       tmin——该列数据的最小值
#       tmax——该列数据的最大值
# 输出：truncated_data——裁剪后得到的数据集
#双边截断
def data_trip(data, index, tmin, tmax):
    # 数据进行双边截断
    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > tmax, index] = tmax
    truncated_data.loc[truncated_data[index] < tmin, index] = tmin

    return truncated_data


# 计算在某个裁剪半径下的误差花销
# 输入：x——预计裁剪半径
#       args——由[data, index, m]组成，其中
#              data——数据集
#              index——求和目标列的索引
#              m——求和目标列数据的均值
# 输出：该裁剪半径下的损失函数值
def compute_tao_spend(x, args):
    tao = x[0]
    data, index, m, epsilon = args
    # 计算累计相差的绝对值
    truncated_data = data.copy()
    sum_max = (truncated_data.loc[truncated_data[index] > int(m + tao), index] - int(m + tao)).sum()
    sum_min = (truncated_data.loc[truncated_data[index] < int(m - tao), index] - int(m - tao)).sum()
    # print(sum_max, sum_min)
    result = abs(sum_max)+abs(sum_min)+ 2 * pow((m+tao)/epsilon,2)
    # print(tao, result)
    return result

# 计算确定裁剪范围的数据求和结果
# 输入：min_range——左阈值
#       max_range——右阈值
#       data——需要进行查询的数据集
#       index——求和目标列的索引
#       m——该列数据的均值
# 输出：将数据裁剪到[min_range,max_range]的范围后的求和值
def compute_range_sum(data, index, min_range, max_range):
    truncated_data = data.copy()
    truncated_data.loc[truncated_data[index] > max_range, index] = max_range
    truncated_data.loc[truncated_data[index] < min_range, index] = min_range
    # print(sum_max, sum_min)
    # print(truncated_data.loc[:, index].sum())
    return truncated_data.loc[:, index].sum()


# 自适应获取裁剪阈值
# 输入：data——需要进行查询的数据集
#       index——求和目标列的索引
#       m——目标列数据的均值
#       tmax——目标列数据的最大值
#       epsilon——隐私预算的大小
# 输出：res.x——最优裁剪阈值大小

def auto_clip(data, index, m, tmax, epsilon):
    x = [1]
    # print("最大bound值:", tmax-m)
    res = optimize.minimize(compute_tao_spend, x, method='COBYLA', args=[data, index, m, epsilon],
                            bounds=[(1, tmax-m)])
    print("最优化函数最小值:", res.fun)
    print("对应tao的取值:", res.x)
    return int(res.x)

# 稀疏向量技术无偏处理
# 输入：data——需要处理的数据集
#       index——求和目标列的索引
#       min_range——左阈值
#       max_range——右阈值

def no_bias(data, index, min_range, max_range):
    bs = range(int(min_range), int(max_range))
    threshold = 10e9
    best_bs = min_range
    # 标准裁剪结果
    standard_sum = data.loc[:, index].sum()
    # print("未裁剪结果:"+str(standard_sum))

    for b in bs:
        r = compute_range_sum(data, index, b, max_range)
        # print("test_min_range="+str(b)+",result="+str(r))

        # if the new answer is pretty close to the old answer, stop
        if abs(r - standard_sum) <= threshold:
            threshold = abs(r-standard_sum)
            best_bs = b

    return best_bs



if __name__ == "__main__":
    # 读取数据
    data = read_data_1()
    index = '每周工作时间'
    tmin, tmax = get_min_max_from_data(data, index)
    print(tmin, tmax)
    m, t = get_avg_std_from_data(data, index)

    # 自适应获取裁剪阈值
    tao = auto_clip(data, index, m, tmax)
    print("自适应裁剪半径:"+str(tao)+",裁剪范围：("+str(max(m-tao,tmin))+","+str(min(m+tao, tmax))+")")

    # 稀疏向量技术无偏处理
    # 如果单边裁剪
    if m-tao<=tmin and m+tao<tmax:
        best_min_range = no_bias(data, index, tmin, m+tao)
        print("无偏处理后的裁剪范围:("+str(best_min_range)+","+str(min(m+tao, tmax))+")")
    elif m-tao>=tmin and m+tao==tmax:
        print("决策为不裁剪，即范围是：("+str(tmin)+","+str(tmax)+")")




