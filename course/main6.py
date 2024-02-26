import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf

# 切片和索引
data = np.arange(24)
print(data)
data = np.arange(24).reshape(2, 3, 4)
print(data)
print(data[0, 0, 1])
print(data[:, 0, 1])  # 钻取
print(data[1])  # 切片
print(data[1, :, :])  # 切片
print(data[1, ...])  # 切片
print(data[0, 1, ::2])  # 间隔选取
print(data[0, :, -1])  # 最后一列
print(data[0, ::-1, -1])  # 反向选取

# 读取 csv 数据文件——loadtxt
ClosePrice, Volume = np.loadtxt(
    "F:\\CS\\Python\\recruitment\\course\\SP500_NOV2019_Hist.csv",
    delimiter=',', usecols=(4, 5), skiprows=1, unpack=True)
print(ClosePrice)
print(Volume)

# VWAP 和 TWAP 计算
vwap = np.average(ClosePrice, weights=Volume)
print(vwap)

t = np.arange(len(ClosePrice))
twap = np.average(ClosePrice, weights=t)
print(twap)

# 矩阵处理
A = np.mat('1 2 3;4,5,6;7,8,9')  # 创建矩阵
print(A)
print(A.T)  # 转置
B = np.matrix([[1, 2], [-1, -3]])
print(B)
print(B.I)  # 逆矩阵
print(B * B.I)
C = np.mat(np.arange(9).reshape(3, 3))  # 更改维数
print(C)

# 利萨如曲线绘制
t = np.linspace(-np.pi, np.pi, 201)
print(t)
a = float(input("请输入第 1 个频率参数："))
b = float(input('请输入第 2 个频率参数：'))
x = np.sin(a * t + np.pi / 2)
y = np.sin(b * t)
plt.plot(x, y)
plt.show()

# 求解线性方程组
A = np.mat('1,-2,1;0,2,-8;-4,5,9')  # 系数
B = np.array([0, 8, -9])  # 常数项
print(A)
print(B)
x = np.linalg.solve(A, B)  # 解向量
print(x)
print(np.dot(A, x))  # 验证——点乘

# 求解特征值和特征向量
A = np.mat('3,-2;1,0')
print(A)
e = np.linalg.eigvals(A)  # 特征值
print(e)
e_vector = np.linalg.eig(A)  # 特征向量
print(e_vector)

# 终值与现值
final_value = npf.fv(0.03 / 4, 5 * 4, -10, -1000)  # 年利率为 3%，计息周期为 1 季度，本金 1000，每季度额外存入 10，存款周期为 5 年
print(final_value)
final_value = []
for i in range(1, 50):
    final_value.append(npf.fv(0.03 / 4, i * 4, -10, -1000))
plt.plot(final_value, 'bo')
plt.show()
present_value = npf.pv(0.03 / 4, 5 * 4, -10, 1376.0963320407982)
print(present_value)

# 布莱克曼窗平滑股价数据
ClosePrice = np.loadtxt(
    "F:\\CS\\Python\\recruitment\\course\\SP500_NOV2019_Hist.csv",
    delimiter=',', usecols=(4,), skiprows=1, unpack=True)
N = int(input('请输入窗函数的窗口长度（日）：'))
window1 = np.blackman(N)
smoothed1 = np.convolve(window1 / window1.sum(), ClosePrice, mode='same')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码

plt.plot(ClosePrice[N:-N], label='原始收盘价')
plt.legend(loc='best')
plt.show()

# 绘制不同的窗函数
window2 = np.bartlett(N)  # 巴特利特窗
window3 = np.hamming(N)  # 汉明窗
window4 = np.kaiser(N, 14)  # 凯泽窗
smoothed2 = np.convolve(window2 / window2.sum(), ClosePrice, mode='same')
smoothed3 = np.convolve(window3 / window3.sum(), ClosePrice, mode='same')
smoothed4 = np.convolve(window4 / window4.sum(), ClosePrice, mode='same')
plt.plot(smoothed1[N:-N], lw=2, label='布莱克曼窗')
plt.plot(smoothed2[N:-N], lw=2, label='巴特利特窗')
plt.plot(smoothed3[N:-N], lw=2, label='汉明窗')
plt.plot(smoothed4[N:-N], lw=2, label='凯泽窗')
plt.plot(ClosePrice[N:-N], label='原始收盘价')
plt.legend(loc='best')
plt.show()

# 经济统计运算
HighPrice, LowPrice, ClossPrice = np.loadtxt(
    "F:\\CS\\Python\\recruitment\\course\\SP500_NOV2019_Hist.csv",
    delimiter=',', usecols=(2, 3, 4), skiprows=1, unpack=True)
print('该区间股价最高价的最大值：', HighPrice.max())
print('该区间股价最低价的最小值：', LowPrice.min())
print("*************************************")
# 近期最高最低价格的极差
print('该区间股价最高价格的极差', np.ptp(HighPrice))
print("该区间股价最低价格的极差", np.ptp(LowPrice))
print("*************************************")
simpleReturn = np.diff(ClossPrice)
print('计算简单收益率：', simpleReturn)
logReturn = np.diff(np.log(ClossPrice))

# 时序数据分析
print("*************************************")


def get_week(date):
    from datetime import datetime
    date = date.decode('utf-8')
    return datetime.strptime(date, "%m/%d/%Y").weekday()


week, ClosePrice = np.loadtxt(
    "F:\\CS\\Python\\recruitment\\course\\SP500_NOV2019_Hist.csv",
    delimiter=',', usecols=(0, 4), converters={0: get_week}, skiprows=1, unpack=True)
allAvg = []
for weekday in range(4):
    average = ClosePrice[week == weekday + 1].mean()
    allAvg.append(average)
    print('星期 %s 的平均收盘价格：%s' % (weekday + 1, average))
print('平均收盘价的最低是星期', np.argmin(allAvg) + 1)
print('平均收盘价的最高是星期', np.argmax(allAvg) + 1)
