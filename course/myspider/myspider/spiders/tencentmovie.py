import scrapy
from ..items import MyspiderItem

class TencentmovieSpider(scrapy.Spider):
    name = "tencentmovie"
    allowed_domains = ["v.qq.com"]
    start_urls = ["https://v.qq.com/channel/net_tv/"]

    def parse(self, response):
        items = MyspiderItem()
        lists = response.xpath('/html/body/div[9]/div/div[2]/div/div/a')

        for i in lists:
            items['name'] = i.xpath('./@title').get()
            items['link'] = i.xpath('./@href').get()

            yield  items
        pass
