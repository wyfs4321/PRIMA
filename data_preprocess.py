import pandas as pd
import pprint
import argparse

data_dir = "./data/"
# epsilon = 1
# max_times_noise = 1000
# n = 100000
# p = 0.01

parser = argparse.ArgumentParser()
parser.add_argument('--source_dimen', default=30, help='源数据值域')
parser.add_argument('--target_dimen', default=10, help='压缩数据值域')

opt, unknown = parser.parse_known_args()
source_dimen = opt.source_dimen
target_dimen = opt.target_dimen

# 功能：读取高维数据集
# 输入：fileName——需要读取的数据集的文件位置
# 输出：df——返回读取的数据集的dataFrame形式
def read_data_high_level(fileName):
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    df = pd.read_csv(fileName, encoding='utf-8')
    return df

# 功能：将高维数据集的值域压缩至10维
# 输入：data_high——需要进行值域压缩的数据
#       source_dimen——原始的值域大小
#       target_dimen——压缩后的值域大小
# 输出：df——返回值域压缩后的数据集
def process_high_to_low(data_high, source_dimen, target_dimen):
    df = data_high.copy()
    for i in range(source_dimen):
        df = df.replace(i,int(i/(source_dimen/target_dimen)))
    return df

if __name__ == "__main__":
    # data_high = read_data_high_level(data_dir+"data_adult_30-50-1M.csv")
    data_high = read_data_high_level('Normal_10-50.csv')
    data_low = process_high_to_low(data_high, source_dimen, target_dimen)

    data_low.to_csv('data_adult_30to10-50-1M.csv', header = 0, index = 0)