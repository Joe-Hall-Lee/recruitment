import json
import random
import time

import requests
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By

# requests 库
r = requests.get('https://data.eastmoney.com/report/')
print(r.status_code)
print(r.headers['content-type'])
print(r.encoding)
print(r.text)

# headers 定制请求头
print(r.request.headers['User-Agent'])
headers = {'User-Agent': 'CQQ'};
r = requests.get('http://www.cntour.cn', headers=headers)
print(r.request.headers['User-Agent'])

# 无定制请求头
r = requests.get('https://movie.douban.com/top250')
print(r.status_code)

# 定制请求头信息
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

r = requests.get('https://movie.douban.com/top250', headers=headers)
print(r.headers)
print(r.status_code)
print(r.text)

# 循环翻页实现
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i * 25)

    r = requests.get(link, headers=headers)
    print(r.status_code)
    print(r.text)

# find 定位标签实现
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i * 25)

    r = requests.get(link, headers=headers)

    soup = BeautifulSoup(r.text, 'html')
    div_list = soup.find_all('div', class_='hd')
    for each in div_list:
        print(each)

# 获取电影名清单
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i * 25)

    r = requests.get(link, headers=headers)

    soup = BeautifulSoup(r.text, 'html')
    div_list = soup.find_all('div', class_='hd')
    for each in div_list:
        movie_name = each.a.span.text.strip()
        print(movie_name)

# css 选择器定位标签实现
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i * 25)

    r = requests.get(link, headers=headers)

    soup = BeautifulSoup(r.text, 'html')
    div_list = soup.select('#content > div > div.article > ol > li > div > div.info > div.hd > a > span:nth-child(1)')
    for each in div_list:
        movie_name = each.text.strip()
        print(movie_name)

# xpath 定位标签实现
for i in range(0, 10):
    link = 'https://movie.douban.com/top250?start=' + str(i * 25)

    r = requests.get(link, headers=headers)
    element_html = etree.HTML(r.text)
    movie_list = element_html.xpath("//div[@class='hd']/a/span[1]")

    for each in movie_list:
        movie_name = each.text.strip()
        print(movie_name)

# 反爬虫技巧，以彼之道还施彼身
header = {
    'Host': 'pearvideo.com',
    'Referer': 'https://pearvideo.com/video_1791325',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# 原始网页的 URL
url = 'https://pearvideo.com/videoStatus.jsp?contId=1791325&mrd=0.9777695152009254'
s = requests.Session()
s.get(url, headers=header)  # 请求首页获取 cookies
cookie = s.cookies  # 为此次获取的 cookies
print(cookie)

html = s.get(url, headers=header, cookies=cookie)

# 解析 JSON 文件
json_data = json.loads(html.text, strict=False)
# 解析 json 文件，后跟中括号为解析的路径
srcUrl = json_data['videoInfo']['videos']['srcUrl']
print(srcUrl)


# 打开浏览器完成相关设置
browser = webdriver.Chrome()
browser.get('https://www.lagou.com/wn/zhaopin')
time.sleep(3)

# 完成搜索点击
browser.find_element(By.CLASS_NAME, 'search-input__1smvz').send_keys("python")  # 定位搜索框输入关键字
browser.find_element(By.CLASS_NAME, 'search-btn__1ilgU').click()  # 点击搜索
browser.maximize_window()  # 最大化窗口
time.sleep(3)

# 爬取内容并解析
browser.execute_script("scroll(0, 3000)")  # 下拉滚动条
items = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')
# 遍历，获取这一页的每条招聘信息
for item in items:
    job_name = item.find_element(By.XPATH, './/*[@id="openWinPostion"]').text
    company_name = item.find_element(By.XPATH, './/div[1]/div[2]/div[1]/a').text
    industry = item.find_element(By.XPATH, './/div[@class="il__3lk85"]').text
    salary = item.find_element(By.XPATH, './/span[@class="money__3Lkgq"]').text
    experience_edu = item.find_element(By.XPATH, './/div[@class="p-bom__JlNur"]').text
    welfare = item.find_element(By.XPATH, './/div[2]/div[2]').text
    job_label = item.find_element(By.XPATH, './/div[2]/div[1]').text
    data = f'{job_name}, {company_name}, {industry}, {salary}, {experience_edu}, {welfare}, {job_label}'
    # 爬取数据，输出
    print(data)

# 翻页继续爬取
for i in range(29):
    browser.find_element(By.CLASS_NAME, 'lg-pagination-next').click()
    time.sleep(2)
    browser.execute_script("scroll(0, 3000)")  # 执行 js 代码下拉滚动条
    # 获取数据
    # 爬取内容并解析
    items = browser.find_elements(By.XPATH, '//*[@id="jobList"]/div[1]/div[@class="item__10RTO"]')
    # 遍历，获取这一页的每条招聘信息
    for item in items:
        job_name = item.find_element(By.XPATH, './/*[@id="openWinPostion"]').text
        company_name = item.find_element(By.XPATH, './/div[1]/div[2]/div[1]/a').text
        industry = item.find_element(By.XPATH, './/div[@class="il__3lk85"]').text
        salary = item.find_element(By.XPATH, './/span[@class="money__3Lkgq"]').text
        experience_edu = item.find_element(By.XPATH, './/div[@class="p-bom__JlNur"]').text
        welfare = item.find_element(By.XPATH, './/div[2]/div[2]').text
        job_label = item.find_element(By.XPATH, './/div[2]/div[1]').text
        data = f'{job_name}, {company_name}, {industry}, {salary}, {experience_edu}, {welfare}, {job_label}'
        # 爬取数据，输出
        print(data)
    time.sleep(random.randint(3, 5))  # 休眠
