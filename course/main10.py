import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
from keras.datasets import mnist, imdb
from keras import models, layers, losses, metrics
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow import optimizers
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
import keras.utils as utils
from sklearn.model_selection import train_test_split
import seaborn as sns



# 神经元激活函数
# 绘制步调函数图像

x = [1, 2, 3, 4]
y = [0, 1, 2 ,3]
plt.step(x, y)
plt.show()
# 设置 sigmoid 函数计算流程
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

mySamples = []
mySigmoid = []
# 设置函数绘制区间
x = np.linspace(-10, 10, 10)
y = np.linspace(-10, 10, 100)
# 在给定区间内绘制 sigmoid 函数值点，形成函数曲线
plt.plot(x, sigmoid(x), 'r', label='linspace(-10, 10, 10')
plt.plot(y, sigmoid(y), 'y', label='linspace(-10, 10, 1000')
plt.grid()
plt.title('Sigmoid function')
plt.suptitle('Sigmoid')
plt.legend(loc='lower right')
# 给绘制曲线图像做标注
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^(-x)}$', fontsize=15)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)
digit = test_images[0]
plt.figure()
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# 神经网络构建
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation="softmax"))
# 神经网络参数
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 数据预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') /255
print("before change: ", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: "), test_labels[0]
# 训练网络
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 测试网络
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_acc: ', test_acc)

# 网络应用
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 将抽取的图片输入网络，并让网络判断图片表示的数值
test_images = test_images.reshape(10000, 28*28)
res = network.predict(test_images)
print(res[1])
for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is", i)
        break

# 加载数据集
# num_words 表示加载影评时，确保影评里面的单词使用频率保持在前 1 万位，于是有些很少见的生僻词在数据加载时会舍弃掉
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])

# 单词内容还原
# 频率与单词的对应关系存储在哈希表 word_index 中，它的 key 对应的是单词，value 对应的是单词的频率
word_index = imdb.get_word_index()
# 我们要把表中的对应关系反转一下，变成 key 是频率，value 是单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
"""
在 train_data 所包含的数值中，数值 1、2、3 对应的不是单词，而用来表示特殊含义，1 表示“填
充”，2 表示“文本起始”，3 表示“未知”，因此当我们从 train_data 中读到的数值是 1、2、3 时，
我们要忽略它，从 4 开始才对应单词，如果数值是 4，那么它表示频率出现最高的单词
"""
text=""
for wordCount in train_data[0]:
    if wordCount > 3:
        text += reverse_word_index.get(wordCount - 3)
        text += " "
    else:
        text += "?"

print(text)

# One-Hot Vector 实现
def oneHotVecterizeText(allText, dimension=10000):
    '''
    allText 是所有文本集合，每条文本对应一个含有 10000 个元素的一维向量，假设文本共有 X 条，那么
    该函数会产生 X 条维度为一万的向量，于是形成一个含有 X 行 10000 列的二维矩阵
    '''
    oneHotMatrix = np.zeros((len(allText), dimension))
    for i, wordFrequence in enumerate(allText):
        oneHotMatrix[i, wordFrequence] = 1.0
    return oneHotMatrix


x_train = oneHotVecterizeText(train_data)
x_test = oneHotVecterizeText(test_data)

print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

texts = ["orange banana apple grape", "banana apple apple", "grape", "orange apple", "banana banana apple banana"]
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.vocabulary_)
print(cv_fit)
print(cv_fit.toarray())


# 实例化 tf 实例
tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
# 输入训练集矩阵，每行表示一个文本
train = ["Chinese Beijing Chinese",
         "Chinese Chinese Shanghai",
         "Chinese Macao",
         "Tokyo Japan Chinese"]

# 训练，构建词汇表以及词项 idf 值，并将输入文本列表转成 VSM 矩阵形式
tv_fit = tv.fit_transform(train)
# 查看一下构建的词汇表
print(tv.get_feature_names_out())
# 查看输入文本列表的 VSM 矩阵
print(tv_fit.toarray())

# 神经网络构建
model = models.Sequential()
# 构建第一层和第二层网络网络，第一层有 10000 个节点，第二层有 16 个节点
# Dense 的意思是，第一层每个节点都与第二层的所有节点相连接
# relu 对应的函数 relu(x) = max(0, x)
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# 第三层有 16 个神经元，第二层每个节点与第三层每个节点都相互连接
model.add(layers.Dense(16, activation='relu'))
# 第四层只有一个节点，输出一个 0-1 之间的概率值
model.add(layers.Dense(1, activation='sigmoid'))

# Relu 激活函数
x = np.linspace(-10, 10)
y_relu = np.array([0*item if item < 0 else item for item in x])

plt.figure()
plt.plot(x, y_relu, label='Relu')
plt.legend()
# 损失函数与链路参数
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 训练神经网络
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 神经网络调参——炼丹
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))

train_result = history.history
# print(train_result.keys())
#
# # 绘制评估曲线
# acc = train_result['accuracy']
# val_acc = train_result['val_accuracy']
# loss = train_result['loss']
# val_loss = train_result['val_loss']
#
# epochs = range(1, len(acc) + 1)
# # 绘制训练数据识别准确度曲线
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # 绘制校验数据识别的准确度曲线
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title("Training and validation loss")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 神经网络应用
# print(model.predict(x_test))

# # 读取数据——pandas 库
# df = pd.read_json("F:/CS/Python/recruitment/course/News_Category_Dataset.json")
# print(df.head(3))
# print(df.columns)
#
# # 查看目标分类（Class）
# categories = df.groupby('category')
# print("total categories: ", categories.ngroups)
# print(categories.size())
#
# # 数据清理——归并重复项
# df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
#
# # 数据转换——单词标号
# df['text'] = df.headline + " "+df.short_description  # 合并标题和摘要
#
# # 将单词进行标号
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df.text)
# X = tokenizer.texts_to_sequences(df.text)
# df['words'] = X
#
# # 数据清理
# # 记录每条数据的单词数
# df['word_length'] = df.words.apply(lambda i: len(i))
# # 清除单词数不足 5 个的数据条目（过短的句子）
# df = df[df.word_length >= 5]
# print(df.word_length.describe())
#
#
# # 数据处理——优化数据维度
# def word2Frequent(sequences):
#     word_index = {}
#     for sequence in sequences:
#         for word in sequence:
#             word_index[word] = word_index.get(word, 0) + 1
#     return word_index
#
#
# word_index = word2Frequent(df.words)
#
# count = 10000
# # 将单词按照频率按照升序排序，然后取出排在第一方位的单词频率
# s = [(k, word_index[k]) for k in sorted(word_index, key=word_index.get, reverse=True)]
# print(s[0])
# # 首先取频度最高的前 `count` 个单词
# sorted_words = sorted(word_index, key=word_index.get, reverse=True)
# top_words = sorted_words[:count]
#
# # 创建一个映射，将单词映射到索引
# frequent_to_index = {word: i for i, word in enumerate(top_words)}
# # 数据处理——因变量（分类）编号
# # 将分类进行编号
# categories = df.groupby('category').size().index.tolist()
# category_int = {}
# int_category = {}
# for i, k in enumerate(categories):
#     category_int.update({k:i})
#     int_category.update({i:k})
#
# df['category2id'] = df['category'].apply(lambda x: category_int[x])
#
# print(df['category2id'])
#
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i in range(len(sequences)):
#         for word in sequences[i]:
#             if frequent_to_index.get(word, None) is not None:
#                 pos = frequent_to_index[word]
#                 results[i, pos] = 1.0
#
#     return results
#
# X = np.array(df.words)
# X = vectorize_sequences(X)
# print(X[0])
#
# Y = to_categorical(df['category2id'].values)
#
# # 将数据分成两部分，80% 用于训练，20% 用于测试
# seed = 29
# x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
#
# # 构建神经网络
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
#
# # 当结果是输出多个分类的概率时，用 softmax 激活函数，它将为 30 个分类提供不同的可能性概率值
# model.add(layers.Dense(len(int_category), activation='softmax'))
#
# # 对于输出多个分类结果，最好的损失函数是 categorical_crossentropy
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 训练神经网络
# # 神经网络调参
# history = model.fit(x_train, y_train, epochs=4, validation_data=(x_val, y_val), batch_size=512)
# # 模型效果评价
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# epochs = range(1, len(loss) +1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validiation acc')
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# plt.legend()
# plt.show()

# 加载数据——加利福利亚房价数据集
data_path = "F:/CS/Python/recruitment/course/housing.csv"

housing = pd.read_csv(data_path)
print(housing.info())
print(housing.head(5))
print(housing.describe())

# 观察数据——直方图
housing.hist(bins=50, figsize=(15, 15))
plt.show()

# 观察数据——非数值型数据
print(housing['ocean_proximity'].value_counts())

total_count = housing['ocean_proximity'].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=total_count.index, y=total_count.values, alpha=0.7)
plt.title("Ocean Proximity Summary")
plt.ylabel("Number of Occurences", fontsize=12)
plt.xlabel("Ocean of Proximity", fontsize=12)
plt.show()

# 数据清理——非数值型数据编码
# 将 ocean_proximity 转换为数值
housing['ocean_proximity'] = housing['ocean_proximity'].astype('category')
housing['ocean_proximity'] = housing['ocean_proximity'].cat.codes
# 将 median_house_value 分离出来作为被预测数据
data = housing.values
train_data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]]
train_value = data[:, [8]]
print(train_data[0])
print(train_value[0])

# 数据转换——归一化
print(train_data)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
print(train_data)

# 构建神经网络
model = models.Sequential()
# 网络包含 3 个连接层，第一层 64 个节点，后面两层含有 128 个节点
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
# 由于网络只需要输出预测价格，因此最后一层含有 1 个节点即可
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


# 神经网络训训练及检验
# 神经网络调参
history = model.fit(train_data, train_value, epochs=200, validation_split=0.2, batch_size=32)
print(history.history.keys)
val_mae_history = history.history['mae']
plt.plot(range(1, len(val_mae_history) + 1), val_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# EMA 实现
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(val_mae_history)

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
