
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy import interpolate
from scipy.integrate import dblquad
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.optimize import fmin
from scipy.optimize import leastsq
from scipy.optimize import root, fsolve

# SciPy 安装
print(scipy.constants.pi)


# 一重积分 quad
def myfunction(x, n, k):
    return k * x ** n  # 定义函数 k 倍 x 的 n 次方


# args 里传入的参数为额外参数 n 的值，即为求 5 倍 x 的 3 次方这一函数的定积分，上限为 1，下限为 2
print(quad(myfunction, 1, 2, args=(3, 5)))
result = []
for i in range(15):
    result.append(quad(myfunction, 0, 2, args=(i, 5))[0])
print(result)
plt.plot(result)
plt.show()


# 一重积分应用
def advertisement_return(t):
    return 1000000 * (math.e ** (0.02 * t))  # 定义广告带来的销售额


revenue = quad(advertisement_return, 0, 12)[0]
print("打完广告的销售额为：", revenue)
profit_added = (revenue - 1000000 * 12) * 0.1
print("打广告带来的新增销售额的利润为：", profit_added)
net_profit = profit_added - 130000
print("广告投资净利润为：", net_profit)


# 二重积分 dblquad


# 二重积分
def f(x, y):
    return 4 - x - y


v, err = dblquad(f, 0, 1, 0, 2)
print(v)


# 二重积分算例
def f(x, y):
    return 3 * (x ** 2) * (y ** 2)


def h(x):
    return 1 - x ** 2


v, err = dblquad(f, 0, 1, 0, h)
print(v)
print(16.0 / 315.0)


# 常微分方程求解——odeint
def diff(y, x):
    return np.array(x)
    # 上面定义的函数在 odeint 里面体现的就是 dy / dx = x


x = np.linspace(1, 10, 100)  # 给出 x 范围
y = odeint(diff, 0, x)  # 设初值为 0，此时 y 为一个数组，元素为不同 x 对应的 y 值
# 也可以直接 y = odeint(lambda y, x: x, 0, x)
plt.plot(x, y[:, 0])  # y 数组（矩阵）的第一列（因为维度相同，plt.plot(x, y) 效果相同）
plt.grid()
plt.show()

# 洛伦兹吸引子
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def lorenz(w, t, r, p, b):
    # 位置矢量 w，三个参数 p, r, b
    x, y, z = w.tolist()
    # 分别计算 dx / dt, dy / dt, dz / dt
    return p * (y - x), x * (r - z) - y, x * y - b * z


t = np.arange(0, 30, 0.02)
initial_val = (0.0, 1.00, 0.0)
track = odeint(lorenz, initial_val, t, args=(10.0, 28.0, 3.0))
X, Y, Z = track[:, 0], track[:, 1], track[:, 2]
ax.plot(X, Y, Z, label='lorenz')
ax.legend()
plt.show()

# 一维插值方法——interp1d
x = np.arange(0, 10)
print(x)
y = np.exp(-x / 3.0)
print(y)
plt.plot(x, y, 'o')
plt.show()
f = interpolate.interp1d(x, y)
xnew = np.arange(0, 9, 0.5)
print(xnew)
ynew = f(xnew)  # use interpolation function returned by interp1d
plt.plot(x, y, 'o', xnew, ynew, '*')
plt.show()

# 一维插值应用
hours = np.arange(1, 13)
temperature = [5, 8, 9, 15, 25, 29, 31, 30, 22, 25, 27, 24]
f = interpolate.interp1d(hours, temperature, 'cubic')
New_Time = np.arange(1, 12, 0.1)
# use interpolation function returned by 'interp1d'
Estimate_temperature = f(New_Time)
plt.plot(hours, temperature, 'o', New_Time, Estimate_temperature, '*')
plt.show()

# 概率统计（stats）

# 抗体滴度为 1: 10, 1: 20, 1: 40, 1: 80, 1: 160, 1: 320, 1: 640, 1: 1280
# 分别对应人数 f 为：4, 3, 10, 10, 11, 15, 14,2
Didu = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
Repeat_f = np.array([4, 3, 10, 10, 11, 15, 14, 2])
Data = Didu.repeat(Repeat_f)
print(Data)
print(Data.mean())
print(scipy.stats.gmean(Data))  # 几何均值
print(scipy.stats.iqr(Data))  # 四分位数间距
print(scipy.stats.variation(Data))  # 变异系数
print(scipy.stats.skew(Data))  # 偏度
print(scipy.stats.kurtosis(Data))  # 峰度

# 参数估计
# 若某市某年 18 岁男生身高服从 μ = 167.7，标准差 δ = 5.3 的正态分布，总体中随机抽样 q = 100 个，每次样本含量 n = 10 人，置信度为 1-0.05
# 总体 δ 未知，小样本，服从 t 分布
g, n, μ, s, a = 100, 10, 167.7, 5.3, 0.05
# 记录每次实验样本的的置信区间
records = []
for i in range(g):
    data = scipy.stats.norm.rvs(loc=167.7, scale=5.3, size=n)
    print(data)
    # 每次样本的值均和标准误
    mean, xsem = data.mean(), scipy.stats.sem(data)
    # 1-a置信区间CI为：
    records.append([mean - scipy.stats.t.isf(a / 2, n - 1) * xsem,
                    mean + scipy.stats.t.isf(a / 2, n - 1) * xsem])
# 置信区间包含住总体μ的概率
count = 0
for i in range(g):
    if records[i][0] < μ < records[i][1]:
        count += 1
print('概率为：', count / g)

# 假设检验
# 完全随机设计实验
# 两样本均来自正态总体
# 两样本的方差相等
# H0: μ1 = μ2，即阿卡波糖胶囊组与拜唐苹胶囊组空腹血糖下降值的总体均数相等
# H0: μ1 = μ2，即阿卡波糖胶囊组与拜唐苹胶囊组空腹血糖下降值的总体均数不相等
# 待测两样本均来自正态总体
data = np.array([[-0.7, -5.6, 2, 2.8, 0.7, 3.5, 4, 5.8, 7.1, -0.5, 2.5, -1.6, 1.7, 3, 0.4, 4.5, 4.6, 2.5, 6, -1.4],
                 [3.7, 6.5, 5, 5.2, 0.8, 0.2, 0.6, 3.4, 6.6, -1.1, 6, 3.8, 2, 1.6, 2, 2.2, 1.2, 3.1, 1.7, -2]])
# equal_var 参数表示，两样本总体方差是否相等，默认为 True
# 方法 1：现有两组样本，直接计算
print(scipy.stats.ttest_ind(data[0], data[1]))

# 优化模块（optimize）
# 拟合 leastsq
# 样本数据(Xi,Yi)，需要转换成数组（列表）形式
Xi = np.array([160, 165, 158, 172, 159, 176, 160, 162, 171])
Yi = np.array([58, 63, 57, 65, 62, 66, 58, 59, 62])


# 需要拟合的函数func：指定函数的形状 k = 0.42116973935 b = -0.828830260655
def func(p, x):
    k, b = p
    return k * x + b


# 偏差函数：x, y 都是列表：这里的 x, y 跟上面的 Xi, Yi 是一一对应的
def error(p, x, y):
    return func(p, x) - y


# k, b 的初始值，可以任意定，经过几次试验，发现 p0 的值会影响 cost 的值：Para[1]
p0 = [1, 20]
Para = leastsq(error, p0, args=(Xi, Yi))
# 读取结果
k, b = Para[0]
print('k=', k, 'b=', b)
# 画样本点
plt.figure(figsize=(8, 6))  # 指定图像比例：8: 6
plt.scatter(Xi, Yi, color='green', label='样本数据', linewidth=2)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
# 画拟合直线
x = np.linspace(150, 190, 100)  # 在 150-190 直接画 100 个连续点
y = k * x + b  # 函数式
plt.plot(x, y, color='red', label='拟合直线', linewidth=2)
plt.legend()
plt.show()


# 最小值求解 fmin
def cost_function(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x_center = np.array([0, 0])
step = 0.5
x0 = np.vstack((x_center, x_center + np.diag((step, step))))
xtol, ftol = 1e-3, 1e-3
xopt, fopt, iter, funcalls, warnflags, allvecs = fmin(
    cost_function, x_center, initial_simplex=x0, xtol=xtol, ftol=ftol, disp=1, retall=1, full_output=1)
print(xopt, fopt)
n = 50
x = np.linspace(-6, 6, n)
y = np.linspace(-6, 6, n)
z = np.zeros((n, n))
for i, a in enumerate(x):
    for j, b in enumerate(y):
        z[i, j] = cost_function([a, b])
xx, yy = np.meshgrid(x, y)
fig, ax = plt.subplots()
c = ax.pcolormesh(xx, yy, z.T, cmap='jet')
fig.colorbar(c, ax=ax)
t = np.asarray(allvecs)
x_, y_ = t[:, 0], t[:, 1]
ax.plot(x_, y_, 'r', x_[0], y_[0], 'go', x_[-1], y_[-1], 'y+', markersize=6)
fig2 = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.plot(x_)
ax1.set_title('x')
ax2.plot(y_)
ax2.set_title('y')
plt.show()

# 方程求解 fsolve
# plt.rc('text', usetex=True)  # 使用 latex
# 使用 scipy.optimize 模块的 root 和 fsolve 函数进行数值求解方程
# 1. 求解 f(x) = 2 * sin(x) - x + 1
rangex1 = np.linspace(-2, 8)
rangey1_1, rangey1_2 = 2 * np.sin(rangex1), rangex1 - 1
plt.figure(1)
plt.plot(rangex1, rangey1_1, 'r', rangex1, rangey1_2, 'b--')
plt.title('$2sin(x)$ and $x-1$')


def f1(x):
    return np.sin(x) * 2 - x + 1


soll_root = root(f1, [2])
soll_fsolve = fsolve(f1, [2])
plt.scatter(soll_fsolve, 2 * np.sin(soll_fsolve), linewidths=9)
plt.show()


# 2. 求解线性方程组 {3x1 + 2x2 = 3; x1 - 2x2 = 5}
def f2(x):
    return np.array([3 * x[0] + 2 * x[1] - 3, x[0] - 2 * x[1] - 5])


sol2_root = root(f2, [0, 0])
sol2_fsolve = fsolve(f2, [0, 0])
print(sol2_fsolve)  #
a = np.array([[3, 2], [1, -2]])
b = np.array([3, 5])
x = np.linalg.solve(a, b)
print(x)  # [ 2.  -1.5]


# 3. 求解非线性方程组
def f3(x):
    return np.array([2 * x[0] ** 2 + 3 * x[1] - 3 * x[2] ** 3 - 7,
                     x[0] + 4 * x[1] ** 2 + 8 * x[2] - 10,
                     x[0] - 2 * x[1] ** 3 - 2 * x[2] ** 2 + 1])


sol3_root = root(f3, [0, 0, 0])
sol3_fsolve = fsolve(f3, [0, 0, 0])
print(sol3_fsolve)


# 非线性方程
def f4(x):
    return np.array(np.sin(2 * x - np.pi) * np.exp(-x / 5) - np.sin(x))


init_guess = np.array([[0], [3], [6], [9]])
sol4_root = root(f4, init_guess)
sol4_fsolve = fsolve(f4, init_guess)
print(sol4_fsolve)
t = np.linspace(-2, 12, 2000)
y1 = np.sin(2 * t - np.pi) * np.exp(-t / 5)
y2 = np.sin(t)
plt.figure(2)
a, = plt.plot(t, y1, label='$sin(2x-\\pi)e^{-x/5}$')
b, = plt.plot(t, y2, label='$sin(x)$')
plt.scatter(sol4_fsolve, np.sin(sol4_fsolve), linewidths=8)
plt.title('$sin(2x-\\pi)e^{-x/5}$ and $sin(x)$')
plt.legend()
