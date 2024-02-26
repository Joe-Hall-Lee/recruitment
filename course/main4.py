import pytesseract
from PIL import Image
from urllib.request import urlopen
# 导入 re 模块
import re
import io
import requests

# match 方法
# 将正则表达式编译成 Pattern 对象，注意 hellow 前面的 r 的意思是“原生字符串”
pattern = re.compile(r'hello')

# 使用 re.match匹配文本，获得匹配结果，无法匹配时将返回 None
result1 = re.match(pattern, "hello")
result2 = re.match(pattern, "hellooCQQ!")
result3 = re.match(pattern, "helo CQQ!")
result4 = re.match(pattern, "hello CQQ!")
# 如果 1 匹配成功
if result1:
    # 使用 Match 获得分组信息
    print(result1.group())
else:
    print("1 匹配失败！")

# 如果 2 匹配成功
if result2:
    # 使用 Match 获得分组信息
    print(result2.group())
else:
    print("2 匹配失败！")

# 如果 3 匹配成功
if result3:
    # 使用 Match 获得分组信息
    print(result3.group())
else:
    print("3 匹配失败！")

# 如果 4 匹配成功
if result4:
    # 使用 Match 获得分组信息
    print(result4.group())
else:
    print("4 匹配失败！")


# search 方法
pattern = re.compile("\d+")
# search 是一次匹配，从任意位置开始，返回的是 match 对象
# 和 match 最大的不同，就是开始的位置不一样，没有查找到，返回 None
result = pattern.search("nnd123tyy4556tre189")
# match 类型，后面的操作和 match 方法是一样的
print(result)
print(type(result))
print(result.group())


# findall 方法
# \d 匹配一个数字
pattern1 = re.compile("\d")
result1 = pattern1.findall("hello 123 567")
print(result)

# \d+ 匹配一个或多个数字，如果是多个数字，则必须连续
pattern2 = re.compile("\d+")
result2 = pattern2.findall("hello 123 567 wor65k6")
print(result2)

# \d{3,} 匹配 3 次或者多次，必须连续
pattern3 = re.compile("\d{3,}")
result3 = pattern3.findall("hello 123 567 wor65k6")
print(result3)

# \d{3} 连续匹配三次
pattern4 = re.compile("\d{3}")
result4 = pattern4.findall("hello 123 567 wor654453k6434")
print(result4)

# \d{1,2} 可以匹配一次，也可以匹配两次，以更多的优先
pattern5 = re.compile("\d{1,2}")
result5 = pattern5.findall("hello 123 567 wor65453k6434")
print(result5)

# re.I 表示忽略大小写，"[a-z]{5} 匹配 a-z 的字母五次
pattern6 = re.compile("[a-z]{5}", re.I)
result6 = pattern.findall("hello 123 567 wor65453k6434")
print(result6)

# \w+ 匹配数字、字母、下划线一次或者多次
pattern7 = re.compile("\w+")
result7 = pattern7.findall("hello 123 567 wor65_453k6434")
print(result7)

# \s+ 匹配空白字符一次或者多次
pattern8 = re.compile("\s+")
result8 = pattern8.findall("hello 123 567 wor65_453k6434")
print(result8)

# \W+ 匹配不是下划线、字母、数字
pattern9 = re.compile("\W+")
result9 = pattern9.findall("hello 123 567 wor65_453k6434")
print(result9)

# [\w\W]+ 匹配所有字符，一次或多次
pattern10 = re.compile("[\w\W]+")
result10 = pattern10.findall("hello 123 567 w￥or65_453k6434")
print(result10)

# [abc]+ 匹配 a 或者 b 或者 c——一次或多次
pattern11 = re.compile("[abc]+")
result11 = pattern11.findall("hello b123 c567 w￥ora65_453k6434")
print(result11)

# [^abc|123]+ 获取不是 abc 或者 123 的字符
pattern12 = re.compile("[^abc|123]+")
result12 = pattern12.findall("hello b123 c567 w￥ora65_453k6434")
print(result12)

# .* 匹配任意字符，除了换行符
pattern13 = re.compile(".*")
result13 = pattern13.findall("hello b123 c567 w￥ora65_45ka6434")
print(result13)

# re.I 表示忽略大小写，"[a-z]{5} 匹配 a-z 的字母五次
pattern10 = re.compile("[a-z]{5}", re.I)
# 只查找字符串在 0-8 之间范围的字符，要前不要后（左闭右开）-->只查找 0、1、2、3、4、5、6、7
result10 = pattern10.findall("hello b123 c567 w￥ora65_483ka6434", 0, 8)
print(result10)


# 正则表达式抓取图片 URL
content = '''<img alt="Python" src="https://profile-avatar.csdnimg.cn/057241cbf4ce4f929511c2470f52cedb_joe_hall_lee.jpg!1" />'''
urls = re.findall('src="(.*?)"', content, re.I|re.S|re.M)
print(urls)

name = urls[0].split('/')[-1]
print(name)

post_data = {
    'username': 'qqchen',
    'password': '密码',
    'user_lb': '教职工'.encode('gb2312')
}

params = {
    'c': 'Login',
    'a': 'login'
}

post_url = 'http://xyfw.xujc.com/login/index.php?c=Login&a=login'

session = requests.Session()

headers = {
    "Host": "xyfw.xujc.com",
    "Origin": "http://xyfw.xujc.com",
    "Referer": "http://xyfw.xujc.com/login/index.php?c=Login&a=login",
    "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

login_page=session.post(post_url, data=post_data, headers=headers, params=params)
print(login_page.status_code)
print(login_page.text)

# 获取图片——Get

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

pic_url = 'https://so.gushiwen.cn/user/login.aspx'
login_page = requests.get(pic_url, headers=headers)
print(login_page.text)

# 获取图片——定位
pic_area_pattern = r'src="/RandCode.ashx"'
img_code = re.search(pic_area_pattern, login_page.text).group(0)
print(img_code)
img_num_pattern = re.compile("/.*ashx")
result = img_num_pattern.findall(img_code)
print(result[0])

# 显示图片
img_url = 'https://so.gushiwen.cn/' + result[0]
img_bytes = urlopen(img_url).read()
data_stream = io.BytesIO(img_bytes)

ImageFile = Image.open(data_stream)
ImageFile.show()

GrayImage = ImageFile.convert('L')
GrayImage.show()

# 二值化图像
threshold = 127
table = []
for n in range(256):
    if n < threshold:
        table.append(0)
    else:
        table.append(1)

BinImage = GrayImage.point(table, '1')
BinImage.show()

captcha = pytesseract.image_to_string(GrayImage).encode('utf-8')
print(captcha)
