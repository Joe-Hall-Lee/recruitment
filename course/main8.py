from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import nan as NaN
from pandas import DataFrame
from scipy import stats

# Series
a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar)

# Series 的创建
obj = pd.Series([4, 7, -5, 3])
print(obj)

obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
print(obj3)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
print(obj4)

# Series 运算
obj2 = pd.Series([4, 7, -5, 3])
index = ['d', 'b', 'a', 'c']
print(obj2)
print(obj2[obj2 > 0])  # 布尔运算过滤
print(obj2 * 2)  # 纯量乘法（非矩阵乘法）
print(np.exp(obj2))  # 数学函数

# Series 控制判断
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print('b' in obj2)
print('e' in obj2)
print(pd.isnull(obj2))
print(pd.notnull(obj2))
print(obj2.isnull())  # Series 自身自带算法

# Series 自动对齐索引
print(obj3 * obj4)

# Series 表头属性——name
print(obj3.name)
print(obj3.index.name)

obj3.name = '人口数'
obj3.index.name = '州名'
print(obj3)

# Series 删除条目
print(obj2.drop('b'))
print(obj2.drop(['a', 'c']))

# Series 切片操作
print(obj2['b'])
print(obj2[1])
print(obj2[2:4])
print(obj2[['b', 'a', 'd']])
print(obj2[[1, 3]])
print(obj2[obj2 > 3])

# Dataframe 创建——传入列表
data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]
print(data)

df = pd.DataFrame(data, columns=['网站', '历史'])
print(df)

# Dataframe 创建——传入 arrays
data = {'Site': ['Google', 'Runoob', 'Wiki'], 'Age': [10, 12, 13]}
df = pd.DataFrame(data)

print(df)

# Dataframe 创建——传入字典
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)

print(df)

# Dataframe 引用
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)

# 返回第一行
print(df.loc[0])
# 返回第二行
print(df.loc[1])

# DataFrame 自建索引
df = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(df)
# 指定索引
print(df.loc["day2"])

# DataFrame 赋值
new = [41, 31, 13]
df['duration'] = new

print(df)

sdata = {'day1': 200, 'day2': 1000}
obj = pd.Series(sdata)

df["duration"] = obj

print(df)
df['demo'] = [1, 2, 3]
print(df)

# DataFrame 重新赋值

df.iloc[1, 1] = 77  # 对具体位置重新赋值
print(df)
df.loc['day3', 'duration'] = 999  # 通过标签和表头来重新赋值
print(df)
df['duration'][df.duration < 100] = 10  # 先判断某行某列满足某个条件，然后赋值
print(df)

# DataFrame 插入——_append 方法
df = df._append(df)
print(df)

# DataFrame 插入——insert 方法
df.insert(1, 'newduration', df['duration'])  # 在第一列插入列名为 newduration 的 df 中的 duration 列
print(df)

# DataFrame 插入——pop + insert
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}
df = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(df)
g = df.pop('duration')  # 弹出 duration 列
print(df)
df.insert(0, 'newduration', g)  # 在第 0 列插入

print(df)

# DataFrame 删除
data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}
df = pd.DataFrame(data, index=["day1", "day2", "day3"])
print(df)
df1 = df.drop('duration', axis=1)  # 删除 duration 列
print(df1)
df2 = df.drop(['day1', 'day2'], axis=0)  # 删除行
print(df2)

# 读取文本文件
# 读取文本文件——txt
data = pd.read_table('F:/CS/Python/recruitment/course/是否晋升.txt', encoding='gbk', header='infer', names=None,
                     index_col=None, dtype=None, engine=None, nrows=None)
print(data)
# 读取文本文件——csv
data = pd.read_table('F:/CS\Python/recruitment/course/是否晋升.csv', sep=',', encoding='gbk', header='infer',
                     names=None,
                     index_col=None, dtype=None, engine=None, nrows=None)
print(data)

# 存储文本文件
data.to_csv('F:/CS/Python/recruitment/course/是否晋升2.txt', sep=',', na_rep='na', columns=None, header=True,
            index=True, index_label='序号', mode='w', encoding='gbk')

data = pd.read_excel('F:/CS/Python/recruitment/course/是否晋升.xlsx', sheet_name=0, header=0, index_col=None,
                     names=None, dtype=None)
print(data)

data.to_excel(excel_writer='F:/CS/Python/recruitment/course/是否晋升2.xlsx', sheet_name='Sheet1', na_rep='na',
              header=True, index=True, index_label='序号')

# 横向合并
df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']], columns=['animal', 'name'])
print(df1)
print(df4)

print(pd.concat([df1, df4], axis=1))
# 纵向合并
df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
print(df2)

print(pd.concat([df1, df2]))
print(pd.concat([df1, df2], ignore_index=True))

df3 = pd.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']], columns=['letter', 'number', 'animal'])
print(df3)
print(pd.concat([df1, df3], sort=False))  # sort=Flase: 列的顺序维持原样，不进行重新排序
print(pd.concat([df1, df3], join='inner'))

# 纵向堆叠——_append
data = pd.DataFrame()
a = [[1, 2, 3], [4, 5, 6]]
data = data._append(a, ignore_index=True)
a = [[7, 8, 9], [10, 11, 12]]
data = data._append(a, ignore_index=True)
print(data)
data = data._append(a, ignore_index=False)
print(data)

# 主键合并
# merge——单主键
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'], 'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'], 'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']})
result = pd.merge(left, right, on='key')

# on 参数传递的 key 作为连接键
print(result)

# merge——复合主键
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'], 'key2': ['K0', 'K1', 'K0', 'K1'], 'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'], 'key2': ['K0', 'K0', 'K0', 'K0'], 'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
result = pd.merge(left, right, on=['key1', 'key2'])
# 同时传入两个 key，此时会进行以 ['key1', 'key2'] 列表的形式进行对应，left 的 keys 列表是：[['K0', 'K0'], ['K0', 'K1'], ['K1', 'K0'], ['K2', 'k1']]，
# left 的 keys 列表是：[['K0', 'K0'], ['K1', 'K0'], ['K1', 'K0'], ['K2', 'k0']]，因此会有 1 个 ['K0', 'K0']、2 个 ['K1', 'K0'] 对应。
print(result)

# merge——左连接、右连接、外连接
result = pd.merge(left, right, how='left', on=['key1', 'key2'])
# 左连接
print(result)

result = pd.merge(left, right, how='right', on=['key1', 'key2'])
# 右连接
print(result)

result = pd.merge(left, right, how='outer', on=['key1', 'key2'])
# 外连接——并集
print(result)
left = pd.DataFrame({'A': [1, 2], 'B': [2, 2]})
right = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 2, 2]})
result = pd.merge(left, right, on='B', how='outer')
print(result)

# join 方法
left = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1']}, index=['a', 'b'])
right = pd.DataFrame({'C': ['C0', 'C1'], 'D': ['D0', 'D1']}, index=['c', 'd'])
print(left)
print(right)

print(left.join(right))  # 默认为左连接
print(left.join(right, how='outer'))

# 两个表中的行索引和列索引重叠，使用 join 方法进行合并，使用参数 on 指定重叠的列名
left = pd.DataFrame({'A': ['A0', 'A1'], 'B': ['B0', 'B1'], 'key': ['k1', 'k2']}, index=['a', 'b'])
right = pd.DataFrame({'C': ['C0', 'C1'], 'D': ['D0', 'D1']}, index=['k1', 'k2'])

print(left)
print(right)
print(left.join(right, on='key', how='left'))

# join 方法——重复列名处理
jk = pd.DataFrame({'A': ['a', 'a1', 'a2'], 'B': ['b', 'b1', 'b2'], 'C': ['c', 'c1', 'c2']})
jk1 = pd.DataFrame({'B': ['ab', 'ab1', 'ab2'], 'E': ['e', 'e1', 'e2'], 'C': ['c', 'c1', 'c2']})

print(jk.join(jk1, lsuffix='_左重复'))  # lsuffix 左侧重复的列名添加后缀名
print(jk.join(jk1, rsuffix='_右重复'))  # rsuffix 右侧重复的列名添加后缀名

# combine_first 合并重叠数据
# 合并重叠数据，用这个方法必须保证它们的行索引和列索引有重叠的部分
left = pd.DataFrame(
    {'A': [np.nan, 'A0', 'A1', 'A2'], 'B': [np.nan, 'B1', np.nan, 'B3'], 'key': ['K0', 'K1', 'K2', 'K3']})
right = pd.DataFrame({'A': ['C0', 'C1', 'C2'], 'B': ['D0', 'D1', 'D2']}, index=[1, 0, 2])

print(left)
print(right)
print(left.combine_first(right))  # 用 right 表去填充 left 表，简单来说，就是索引对索引的数据进行填充

# groupby 方法
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

print(df)

print(df.groupby('A'), type(df.groupby('A')))
a = df.groupby('A')[['C', 'D']].mean()
b = df.groupby(['A', 'B']).mean()
c = df.groupby('A')[['D']].mean()  # 以 A 分组，算 D 的平均值
print('----------------')
print(a, type(a), '\n', a.columns)
print(b, type(b), '\n', b.columns)
print(c, type(c))

# drop_duplicates 去重
dict = {'x': [1, 2, 3, 6], 'y': [1, 4, 1, 1], 'z': [1, 2, 4, 1]}
df = pd.DataFrame(dict)
print(df)

df.drop_duplicates(subset=['y', 'z'], keep='first', inplace=True)
print(df)

# Corr 去重
a = np.arange(1, 10).reshape(3, 3)
data = pd.DataFrame(a, index=["a", "b", "c"], columns=["one", "two", "three"])
print(data)

# 计算第一列和第二列的相关系数
print(data.one.corr(data.two))

# Equals 去重
idx1 = pd.Index(['Labrador', 'Beagle', 'Labrador', 'Lhasa', 'Husky', 'Beagle'])
idx2 = pd.Index(['Larador', 'Beagle', 'Pug', 'Lhasa', 'Husky', 'Pitbull'])
print(idx1, "\n", idx2)
print(idx1.equals(idx1))

# Isnull 查找缺失值
df = pd.DataFrame(np.random.randn(10, 6))
df.iloc[1:3, 1] = np.nan
df.iloc[5, 3] = np.nan
df.iloc[7:9, 5] = np.nan
print(df)
print(df.isnull())
print(df.isnull().any())  # 判断哪些“列”存在缺失值
print(df[df.isnull().values == True])  # 只显示有缺失值的行

# dropna 删除缺失值
print(df.dropna(how='all'))  # 传入这个参数后将只丢弃全为缺失值的那些行
print(df.dropna(axis=1))  # 丢失有缺失值的列（一般不会这么做，这样会删掉一个特征）
print(df.dropna(axis=1, how="all"))  # 丢弃全为缺失值的那些列
print(df.dropna(axis=0, subset=[1, 2]))  # 丢弃第 2、3 列有确实值的行

# fillna 替换缺失值
df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])
print(df1)

print(df1.fillna(100))  # 直接常数填充
print(df1.fillna({0: 10, 1: 20, 2: 30}))  # 通过字典填充不同常数
print(df1.fillna(0, inplace=True))  # 直接修改原对象

# interpolate 方法插值
data = pd.Series([1, 2, 3, np.nan, np.nan, 6, np.nan])
print(data)
print(data.interpolate)
print(data.interpolate(method='pad', limit=1))

# 拉依达法则实现
data = [1222, 87, 77, 92, 68, 88, 78, 84, 77, 81, 80, 80, 77, 92, 86,
        876, 80, 81, 75, 77, 72, 81, 72, 84, 86, 80, 68, 77, 87,
        976, 77, 78, 92, 75, 80, 78, 123, 3, 1223, 1232, 1212123213]

df = pd.DataFrame(data, columns=['value'])
print(df)

u = df['value'].mean()
std = df['value'].std()

print(stats.kstest(df['value'], 'norm', (u, std)))  # 此时，pvalue < 0.05，不拒绝原假设，因此上面的数据不服从正态分布
print('均值为：%.3f，标准差为：%.3f' % (u, std))

error = df[np.abs(df['value'] - u) > 3 * std]  # 识别异常值
data_c = df[np.abs(df['value'] - u) <= 3 * std]  # 剔除异常值，保留正常的数据
print(error)

# boxplot 方法
df = DataFrame(np.random.randn(10, 2), columns=['Col1', 'Col2'])
boxplot = df.boxplot()
plt.show()

# get_dummies 处理哑变量
df = pd.DataFrame([['green', 'A'], ['red', 'B'], ['blue', 'A']])
df.columns = ['color', 'class']
print(pd.get_dummies(df))
print(pd.get_dummies(df.color))
print(df.join(pd.get_dummies(df.color)))

# Timestamp 类
p1 = pd.Timestamp(2017, 6, 19)
p2 = pd.Timestamp(dt(2017, 6, 19,
                     hour=9, minute=13, second=45))
p3 = pd.Timestamp("2017-6-19 9:13:45")

# to_datetime 方法
p4 = pd.to_datetime("2017-6-19 9:13:45")
p5 = pd.to_datetime(dt(2017, 6, 19, hour=9, minute=13, second=45))
print("type of p4:", type(p4))
print(p4)
print("type of p5:", type(p5))
print(p5)

# Timedelta 类
print(pd.to_datetime('2019-9-4') - pd.to_datetime('2018-1-1'))
print(pd.Timedelta('3 days 3 hours 3 minutes 30 seconds'))
print(pd.Timedelta(5, unit='d'))

age = (dt.now() - pd.to_datetime('1993-5-27')) / pd.Timedelta(days=365)
print(age)  # 计算生日为 1993 年 5 月 27 日的人的今年的年龄
