import base64
import re
import time
import urllib
from io import BytesIO

import cv2
import numpy
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By

# 初始化
url = 'https://captcha1.scrape.center/'
email = '邮箱'
password = '密码'
browser = webdriver.Chrome()
browser.get(url)
time.sleep(3)

browser.find_element(By.XPATH, "//input[@type='text']").send_keys(email)  # 定位输入用户名
browser.find_element(By.XPATH, "//input[@type='password']").send_keys(password)  # 定位输入密码
time.sleep(3)

browser.find_element(By.XPATH, "//button[@class='el-button el-button--primary']").click()
time.sleep(3)

# 获取背景图片和滑块图片
js_hide_slice = 'document.getElementsByClassName("geetest_canvas_slice")[0].style.display="none"'
browser.execute_script(js_hide_slice)
# 截取缺口图
gap_img_element = browser.find_element(By.CLASS_NAME, 'geetest_canvas_bg')
gap_img_base64 = gap_img_element.screenshot_as_base64
BackgroundImage = Image.open(BytesIO(base64.b64decode(gap_img_base64)))
BackgroundImage.show()

js_hide_gap = 'document.getElementsByClassName("geetest_canvas_bg")[0].style.display="none"'
js_show_slice = 'document.getElementsByClassName("geetest_canvas_slice")[0].style.display="block"'
browser.execute_script(js_hide_gap + ';' + js_show_slice)

slipe_img_element = browser.find_element(By.CLASS_NAME, 'geetest_canvas_slice')
slipe_img_base64 = slipe_img_element.screenshot_as_base64

SlideImage = Image.open(BytesIO(base64.b64decode(slipe_img_base64)))
# 截取图像的前 60 px 宽度
SlideImage = SlideImage.crop((3, 0, 50, SlideImage.height))
SlideImage.show()
js_show_gap = 'document.getElementsByClassName("geetest_canvas_bg")[0].style.display="block"'
browser.execute_script(js_show_gap)

# 获取图片边缘灰度图
BackgroundImageMatrix = numpy.asarray(BackgroundImage)
SlideImageMatrix = numpy.asarray(SlideImage)

Background_edge = cv2.Canny(BackgroundImageMatrix, 100, 200)
Slide_edge = cv2.Canny(SlideImageMatrix, 100, 200)
Background_edge_matrix = cv2.cvtColor(Background_edge, cv2.COLOR_GRAY2RGB)
Slide_edge_matrix = cv2.cvtColor(Slide_edge, cv2.COLOR_GRAY2RGB)
BackGroundEdge_pic = Image.fromarray(Background_edge_matrix)
SlideEdge_pic = Image.fromarray(Slide_edge_matrix)
BackGroundEdge_pic.show()
SlideEdge_pic.show()


# 缺口匹配
fit = cv2.matchTemplate(Background_edge_matrix, Slide_edge_matrix, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(fit)  # 寻找最优匹配
X = max_loc[0]  # 因为是滑条所以只需要 x 坐标
# 绘制方框
th, tw = Slide_edge_matrix.shape[:2]
tl = max_loc  # 左上角点的坐标
br = (tl[0] + tw, tl[1] + th)  # 右下角点的坐标
cv2.rectangle(BackgroundImageMatrix, tl, br, (0, 0, 255), 2)  # 绘制矩形
Fit_pic = Image.fromarray(BackgroundImageMatrix)


Fit_pic.show()
def get_tracks(distance, seconds, ease_func):
    tracks = [0]
    offsets = [0]
    for t in numpy.arange(0.0, seconds, 0.1):
        ease = globals()[ease_func]
        offset = round(ease(t / seconds) * distance)
        tracks.append(offset - offsets[-1])
        offsets.append(offset)
    return offsets, tracks


def ease_out_bounce(x):
    n1 = 7.5625
    d1 = 2.75
    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * x * x + 0.984375


offsets, tracks = get_tracks(X - 2, 8, 'ease_out_bounce')
SlideButton = browser.find_element(By.CLASS_NAME, 'geetest_slider_button')
ActionChains(browser).click_and_hold(SlideButton).perform()

for x in tracks:
    ActionChains(browser).move_by_offset(x, 0).perform()
time.sleep(0.5)
ActionChains(browser).release().perform()

# 确定清楚请求包内容
URL = 'https://pearvideo.com/videoStatus.jsp?contId=1791325&mrd=0.9777695152009254'
headers = {
    'Host': 'pearvideo.com',
    'Referer': 'https://pearvideo.com/video_1791325',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

query_paramters = {
    'contId': '1791325',
    'mrd': '0.9777695152009254'
}

html = requests.get(URL, headers=headers, params=query_paramters)
print(html.text)

video_url = html.json()['videoInfo']['videos']['srcUrl']
print(video_url)

pattern = r"\d+-"

need_replace = re.findall(pattern, video_url)[0]
print(need_replace)

# 替换的字符串
replaced = 'cont-1791325-'
# 真实的下载地址
down_url = video_url.replace(need_replace, replaced)
print(down_url)

urllib.request.urlretrieve(down_url, 'F:/CS/Python/recruitment/course/haha.mp4')

print('ok')
