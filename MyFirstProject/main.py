import time

# list——列表
list = [1, 2, 3, 4, "文本元素", [1, 2, 3]]
print(list)
print(list[0])
list[0] = 7
print(list[0])
print(list[3])
list.remove(3)
print(list[3])

# tuple——元组（元素不可修改）
tuple = (1, 2, 3, 4, "文本元素", [1, 2, 3])
print(tuple)
print(tuple[5])

# dictinoary——字典
information = {'name': 'liming', 'age': '24'}
print(information)
information['sex'] = 'boy'
print(information)
del information['age']
print(information)

# 条件——if
password = input()
if password == '123':
    print('login success')
else:
    print('wrong password')

password = input()
if password == 'abc':
    print('login success acount abc')
else:
    if password == 'xyz':
        print('login success about xyz')
    else:
        print('wrong password')

# 循环——for
sum = 0
for i in range(1, 10, 1):
    sum = i + sum
    print(i, sum)

# 天下武功，唯快不破
# 算法 1
if __name__ == '__main__':
    start_time = time.time()
    for a in range(0, 1001):
        for b in range(0, 1001):
            for c in range(0, 1001):
                if a ** 2 + b ** 2 == c ** 2 and a + b + c == 1000:
                    print("a, b, c分别为：%d, %d, %d" % (a, b, c))

    end_time = time.time()
    print("总耗时：%f" % (end_time - start_time))
    print("算法结束！")

# 算法 2
if __name__ == '__main__':
    start_time = time.time()
    for a in range(0, 1001):
        for b in range(0, 1001):
            c = 1000 - a - b
            if a ** 2 + b ** 2 == c ** 2:
                print("a, b, c分别为：%d, %d, %d" % (a, b, c))

    end_time = time.time()
    print("总耗时：%f" % (end_time - start_time))
    print("算法结束！")

# 人尽其才，物尽其用
# 算法 1
if __name__ == '__main__':
    a = (int)(input("请输入 a 存储的自然数："))
    b = (int)(input("请输入 b 存储的自然数："))

    print("交换前的a和b分别为：%d, %d" % (a, b))

    c = a
    a = b
    b = c
    print("交换后的a和b分别为：%d, %d" % (a, b))

# 算法 2
if __name__ == '__main__':
    a = (int)(input("请输入 a 存储的自然数："))
    b = (int)(input("请输入 b 存储的自然数："))

    print("交换前的 a 和 b 分别为：%d, %d" % (a, b))

    a = a + b
    b = a - b
    a = a - b
    print("交换后的 a 和 b 分别为：%d, %d" % (a, b))


# 函数


def y(x):
    y = 5 * x + 2
    return y


d = y(5)
print(d)

print(y(int(input())))
