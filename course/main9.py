import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn import preprocessing, datasets, decomposition, neighbors, svm, linear_model, metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_swiss_roll, make_blobs, make_circles, load_iris, fetch_california_housing, load_digits
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, \
    QuantileTransformer, Normalizer, minmax_scale, PolynomialFeatures, label_binarize
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC
from scipy.ndimage import convolve
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, SGDClassifier
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from time import time
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from scipy import interpolate

np.random.seed(5)

# scale 标准化
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])

print(X_train.mean(axis=0))
print(X_train.std(axis=0))

X_scaled = preprocessing.scale(X_train)
print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# StandardScaler 标准化
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.transform(data))
print(scaler.transform([[2, 2]]))

# 不同缩放器标准化效果比较
dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target

# 仅采用 2 个特征即可简化可视化
# 0 特征——家庭中位数收入 MedInc: median income in block group 具有长尾分布。
# 特征 5——家庭数 AveOccup: average number of household members 有一些但很大的离群值。

X = X_full[:, [0, 5]]

distributions = [
    ('未缩放原始数据', X),
    ('标准缩放器 Standard Scaler 缩放后数据', StandardScaler().fit_transform(X)),
    ('极值缩放器 MinMaxScaler 缩放后数据', MinMaxScaler().fit_transform(X)),
    ('最大绝对值缩放器 MaxAbsScaler 缩放后数据', MaxAbsScaler().fit_transform(X)),
    ('鲁棒缩放器 RobustScaler 缩放后数据', RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('幂转换器 PowerTransformer——Yeo-Johnson 方法缩放后数据', PowerTransformer(method='yeo-johnson').fit_transform(X)),
    ('幂转换器 PowerTransformer——Box_Cox 方法缩放后数据', PowerTransformer(method='box-cox').fit_transform(X)),
    ('分位数转换器（高斯）QuantileTransformer (Gaussian) 转换后数据',
     QuantileTransformer(output_distribution='normal').fit_transform(X)),
    ('分位数转换器（均匀）QuantileTransformer (uniform) 转换后数据',
     QuantileTransformer(output_distribution='uniform').fit_transform(X)),
    ('归一化 Normalizer 转换后数据', Normalizer().fit_transform(X))
]

# 将输出范围缩放到 0 到 1 之间
y = minmax_scale(y_full)

# matplotlib < 1.5 中不存在 plama 列（直译：血浆那一列）
cmap = getattr(cm, 'plasma_r', cm.hot_r)


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # 定义第一个绘图的轴
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # 定义放大图的轴
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # 定义颜色条的轴
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx), (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom), ax_colorbar)


def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes
    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # 点状图
    colors = cmap(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

    # 移除顶部和右侧脊柱以达到美观
    # 制作漂亮的轴布局
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # 轴 X1 的直方图（功能 5）
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal', color='grey', ec='grey')
    hist_X1.axis('off')

    # 轴 X0 的直方图（功能 0）
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical', color='grey', ec='grey')
    hist_X0.axis('off')


def make_plot(item_idx):
    title, X = distributions[item_idx]
    ax_zoom_out, ax_zoon_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoon_in)
    plot_distribution(axarr[0], X, y, hist_nbins=200, x0_label="收入中位数", x1_label="家庭数", title="完整数据集")

    # 放缩
    zoom_in_precentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_precentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_precentile_range)

    non_outliers_mask = (
            np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
            np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
    plot_distribution(axarr[1], X[non_outliers_mask], y[non_outliers_mask],
                      hist_nbins=50,
                      x0_label="收入中位数",
                      x1_label="家庭数",
                      title="仅考虑数据集的 99%（放大以显示没有边缘异常值的数据集）")

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cmap, norm=norm, orientation='vertical',
                              label='Color mapping for values of y')

    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.show()


make_plot(0)
make_plot(1)
make_plot(2)
make_plot(3)
make_plot(4)
make_plot(5)
make_plot(6)
make_plot(7)
make_plot(8)
make_plot(9)

# 类别特征编码
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.transform([['female', 'from US', 'uses Safari']]))
# 避免编码有序性——独热编码
enc = preprocessing.OneHotEncoder()
enc.fit(X)
print(enc.transform([['female', 'from US', 'uses Safari']]))
print(enc.transform([['female', 'from US', 'uses Safari']]).toarray())

# 离散化
X = np.array([[-3., 5., 15], [0., 6., 14], [6., 3., 11]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)
print(est.transform(X))

# 特征提取
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.0}
]
vec = DictVectorizer()

print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names_out())

# 在 irs 数据集上应用 PCA
# 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 建模降维
pca = decomposition.PCA(n_components=2)  # 降到二维
pca.fit(X)
X = pca.transform(X)
# 作图
plt.figure(figsize=(6, 4))

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    plt.scatter(X[y == label, 0], X[y == label, 1], label=label)
    plt.xlabel('sepal_len')
    plt.ylabel('sepal_wid')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# 在 iris 数据集上应用 k-means
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1, init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clustors, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
    fig.add_axes(ax)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
fig.add_axes(ax)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth.')
ax.dist = 12

plt.show()


# 在 iris 数据集上应用层次聚类
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dengrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of points if no parenthesis).")
plt.show()

# 在 iris 数据集上应用 KNN
n_neighbors = 15

# 导入需要处理的数据
iris = datasets.load_iris()

# 我们仅采用前两个特征。我们可以通过使用二维数据集来避免使用复杂的切片
X = iris.data[:, :2]
y = iris.target

h = .02  # 设置网格中的步长

# 提取色谱
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
for weights in ['uniform', 'distance']:
    # 我们创建最近邻分类器的实例并拟合数据。
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    # 绘制决策边界。为此，我们将为网格 [x_min, x_max] × [y_min, y_max] 中的每个点分配颜色。
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入颜色图
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练数据
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s" % (n_neighbors, weights))

plt.show()

# 支持向量机：最大边际分割超平面
# 我们创建 40 个用来分割的数据点
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# 拟合模型，并且为了展示作用，并不进行标准化
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 绘制 decision function 的结果
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创造网格来评估模型
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和边际
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# 绘制支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
           edgecolors='k')
plt.show()

# 产生样本数据
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# 拟合支持向量机
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)


# 画二维 SVC 决策边界
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 创建评估模型的网络
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # 画决策边界和边界
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 画支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()

X, y = make_circles(100, factor=.1, noise=.1)

r = np.exp((X ** 2).sum(1))


def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')


plot_3D()
plt.show()

# 三种不同类型的 SVM 核函数
# 我们的数据集与标签
X = np.c_[(.4, -.7),
(-1.5, -1),
(-1.4, -.9),
(-1.3, -1.2),
(-1.1, -.2),
(-1.2, -.4),
(-.5, 1.2),
(-1.5, 2.1),
(1, 1),
    # --
(1.3, .8),
(1.2, .5),
(.2, -2),
(.5, -2.4),
(.2, -2.7),
(0, -2.7),
(1.3, 2.1)].T
Y = [0] * 8 + [1] * 8
# 图像的编号
fignum = 1

# 拟合模型
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # 绘制直线，点和最接近平面的向量
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10,
                edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # 将结果放入颜色图
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()


# 在 iris 数据集上应用 SVM
def make_meshgrid(x, y, h=.02):
    """创建要绘制的点网络
    参数
    ------
    x: 创建网格 x 轴所需要的数据
    y: 创建网格 y 轴所需要的数据
    h: 网格大小的可选大小，可选填
    返回
    ------
    xx, yy: n 维数组
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """绘制分类器的决策边界。

    参数
    -----------
    ax: matplotlib 子图对象
    clf: 一个分类器
    xx: 网状网格 meshgrid 的 n 维数组
    yy: 网状网格 meshgrid 的 n 维数组
    params: 传递给 contourf 的参数字典，可选填
    """

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 导入数据以便后续使用
iris = datasets.load_iris()
# 采用前两个特征，我们可以通过使用二维数据集来避免使用切片。
X = iris.data[:, :2]
y = iris.target

# 我们创建一个 SVM 实例并拟合数据。由于要绘制支持向量，因此我们不缩放数据
C = 1.0  # SVM 正则化参数
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in models)

# 为图像设置标题
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polymonial (degree 3) kernel')

# 设置一个 2 × 2 结构的画布
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

# 使用标签传播学习复杂的结构
# 创建带有内圈的环型数据集
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)
outer, inner = 0, 1
labels = np.full(n_samples, -1.)
labels[0] = outer
labels[-1] = inner

########################################################
# 使用 LabelSpreading 学习
label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
########################################################
# 绘制输出标签
output_labels = label_spread.transduction_
plt.subplot(1, 2, 1)
plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy', marker='s', lw=0, label='outer labeled', s=10)
plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c', marker='s', lw=0, label='inner labeled', s=10)
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange', marker='.', lw=0, label='unlabeled')
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Raw data (2 classes=outer and inner)")

plt.subplot(1, 2, 2)
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy', marker='s', lw=0, s=10, label='outer learned')
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c', marker='s', lw=0, s=10, label='inner learned')
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Labels learned with Label Spreading (KNN)")

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
plt.show()

# iris 数据集上的标签传播与 SVM 的决策边界比较
rng = np.random.RandomState(0)

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 网格中的步长
h = .02

y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1
# 我们创建一个 SVM 实例并拟合数据。由于要绘制支持向量，因此我们不缩放数据
ls30 = (LabelSpreading().fit(X, y_30), y_30)
ls50 = (LabelSpreading().fit(X, y_50), y_50)
ls100 = (LabelSpreading().fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)

# 创建要绘制的网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 图像的标题
titles = ['Label Spreading 30% data', 'Label Spreading 50% data', 'Label Spreading 100% data', 'SVC with rbf kernel']

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    # 绘制决策边界。
    # 为此，我们将为网格 [x_min, x_max] × [y_min, y_max] 中的每个点分配颜色。
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入颜色图
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # 绘制训练点
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')

    plt.title(titles[i])

plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()

# 分层聚类：结构化区域与非结构化区域
##############################################################################
# Generate data (swiss roll dataset)
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise=noise)
# Make it thinner
X[:, 1] *= .5
##############################################################################
# Compute clustering
print("Compute unstructured hierachical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)
##############################################################################
# Plot result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(7, -80)
for l in np.unique(label):
    print(X[label == l, 0])
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2], color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')

plt.title("Without connectivity constraints (time %.2f)" % elapsed_time)
##############################################################################
# Define the structure A of the data. Here a 10 nearest neighbors
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

##############################################################################
# Computer Clusteing
print("Compute structured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)
##############################################################################
# Plot result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2], color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
plt.title("With connectivity constraints (time %.2f)" % elapsed_time)

plt.show()


# 用于数字分类的 RBM
##############################################################################
# Setting up
def nudge_dataset(X, Y):
    """
    将 X 中的 8 × 8 图像向左、向右、向下、向上移动 1 px
    这样产生的数据集比原始数据集大 5 倍
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


# Load Data
X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, 'float32')
X, Y = nudge_dataset(X, y)
X = (X - np.min(X, 0) / (np.max(X, 0) + 0.0001))  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 我们要使用的模型
logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)]
)
##############################################################################
# 训练

# 超参数，这些是使用 GridSearchCV 通过交叉验证设置的。
# 在这里，我们不执行交叉验证以节省时间。

rbm.learning_rate = 0.06
rbm.n_iter = 10

# 更多的组件倾向于提供更好地预测性能，但拟合时间更长
rbm.n_components = 100
logistic.C = 6000

# 培训 RBM-Logistic 管道
rbm_features_classifier.fit(X_train, Y_train)

# 直接在像素上训练 Logistic 回归分类器
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.
raw_pixel_classifier.fit(X_train, Y_train)

##############################################################################
# 评估

Y_pred = rbm_features_classifier.predict(X_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)
))

Y_pred = raw_pixel_classifier.predict(X_test)
print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)
))

##############################################################################
# Plotting
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()


# 欠拟合与过拟合
def true_fun(X):
    return np.cos(1.5 * np.pi * X)


np.random.seed(0)
n_samples = 30
degrees = [1, 4, 15]
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)

    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolors='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/-{:.2e})".format(degrees[i], -scores.mean(), scores.std()))

plt.show()

# 糖尿病数据集上的交叉验证
X, y = datasets.load_diabetes(return_X_y=True)
X = X[:150]
y = y[:150]
lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# 显示误差线，显示 +/- 标准。分数错误
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 控制填充颜色的半透明性
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV scores +/- std errors')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])
##############################################################################
# 奖励：您对 alpha 的选择有多少信心呢？
'''
为了回答这个问题，我们使用了 LassoCV 对象，该对象通过内部交叉验证自动从数据中设置其 alpha 参数（即，它对收到的训练数据执行交叉验证）。

我们使用外部交叉验证来查看自动获得的字母在不同交叉验证折痕之间的差异。
'''
lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
k_fold = KFold(3)

print('Answer to the bonus question: ', 'how much can you trust the selection of alpha?')
print()
print('Alpha parameters the generalization score on different')
print('subsets of the data')

for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score:{2:.5f}".format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))

print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially")
plt.show()

# 比较随机搜索和网格搜索
# 获得一些数据
X, y = load_digits(return_X_y=True)
# 建立一个分类器
clf = SGDClassifier(loss='hinge', penalty='elasticnet', fit_intercept=True)


# 实用功能呈现最佳成绩
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validition score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],
                                                                         results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# 指定要从中采样的参数和分布
param_dist = {'average': [True, False], 'l1_ratio': stats.uniform(0, 1), 'alpha': loguniform(1e-4, 1e0)}

# 运行随机搜索
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# 对所有参数使用完整的网格搜索
param_grid = {'average': [True, False], 'l1_ratio': np.linspace(0, 1, num=10),
              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
# 运行网格搜索
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (
time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

# 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将输出二值化
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# 增加一些噪音特征让问题变得更难一些
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# 打乱并分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
# 学习，预测
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# 计算每个类别的 ROC 曲线和 AUC 面积
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算 ROC 曲线和 AUC 面积的微观平均（micro-averaging）
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()

lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc='lower right')
plt.show()
