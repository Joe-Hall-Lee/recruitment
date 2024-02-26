import re
import time
import hashlib
from urllib.parse import urlencode
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad
import requests
from bs4 import BeautifulSoup
import base64
import json

# get 抓取
url = 'https://data.eastmoney.com/report/'

strhtml = requests.get(url)
print(strhtml.text)

# 读取所有头条标题
soup = BeautifulSoup(strhtml.text, 'lxml')
data = soup.select('body > div.main > div.centerbox > div.dlable > a')

print(data)

# 清洗数据
for item in data:
    result = {
        'title': item.get_text(),
        'link': item.get('href'),
        # 正则表达式
        'id': re.findall('\d+', item.get('href'))
    }
    print(result)

# 中文文本清理
url = 'https://data.eastmoney.com/report/info/AP202401171617592379.html'

strhtml = requests.get(url);
strhtml.encoding = 'utf-8'
soup = BeautifulSoup(strhtml.text, 'lxml').find('div', id='ContentBody')
text = soup.text.replace('\n' * 2, '')
text = soup.text.replace(' ' * 2, '')
print(text)

# 自行设计对接接口
url = 'https://dict.youdao.com/webtranslate'

class AESCipher(object):
    key = b'ydsecret://query/key/B*RGygVywfNBwpmBaZg*WT7SIOUP2T0C9WHMZN39j^DAdaZhAnxvGcCY6VYFwnHl'
    iv = b'ydsecret://query/iv/C@lZe2YzHtZ2CYgaXKSVfsb7Y4QWHjITPPZ0nQp87fBeJ!Iv6v^6fvi2WN@bYpJ4'
    iv = hashlib.md5(iv).digest()
    key = hashlib.md5(key).digest()

    @staticmethod
    def decrypt(data):
        # AES 解密
        cipher = AES.new(AESCipher.key, AES.MODE_CBC, iv=AESCipher.iv)
        decrypted = cipher.decrypt(base64.b64decode(data, b'-_'))
        unpadded_message = unpad(decrypted, AES.block_size).decode()
        return unpadded_message

    @staticmethod
    def encrypt(plaintext: str):
        # AES 加密
        cipher = AES.new(AESCipher.key, AES.MODE_CBC, iv=AESCipher.iv)
        plaintext = plaintext.encode()
        padded_message = pad(plaintext, AES.block_size)
        encrypted = cipher.encrypt(padded_message)
        encrypted = base64.b64encode(encrypted, b'-_')
        return encrypted


def get_form_data(sentence):
    """
    构建表单参数
    :param: sentence: 翻译内容
    :return:
    """
    e = 'fsdsogkndfokasodnaso'
    d = 'fanyideskweb'
    u = 'webfanyi'
    m = 'client,mysticTime,product'
    p = '1.0.0'
    b = 'web'
    f = 'fanyi.web'
    t = time.time()

    query = {
        'client': d,
        'mysticTime': t,
        'product': u,
        'key': e
    }

    # 获取 sign 值——密钥值
    h = hashlib.md5(urlencode(query).encode('utf-8')).hexdigest()

    form_data = {
        'i': sentence,
        'from': "AUTO",
        'to': "AUTO",
        'domain': 0,
        'dictResult': 'true',
        'keyid': u,
        'sign': h,
        'client': d,
        'product': u,
        'appVersion': p,
        'vendor': b,
        'pointParam': m,
        'mysticTime': t,
        'keyfrom': f
    }
    return form_data


def spider(sentence):
    """
    :param sentence: 需翻译的句子
    :return:
    """
    # 有道翻译网页请求参数

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'referer': 'https://fanyi.youdao.com/',
        'cookie': 'OUTFOX_SEARCH_USER_ID=-805044645@10.112.57.88; OUTFOX_SEARCH_USER_ID_NCOO=818822109.5585971;'
    }
    params = get_form_data(sentence)

    response = requests.post(url, headers=headers, data=params)
    print(response.text)
    # 翻译结果进行 AES 解密
    cipher = AESCipher
    content = json.loads(cipher.decrypt(response.text))
    print(content)

    print('翻译结果：');
    print(content['translateResult'][0][0]['tgt'])



i = input("请输出要翻译的内容：")
spider(i)
