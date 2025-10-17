#教育机构 ：马士兵教育
#讲    师：杨淑娟
import  requests
from lxml import  etree
url='https://www.qidian.com/rank/yuepiao'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}

#发送请求
resp=requests.get(url,headers)
e=etree.HTML(resp.text)  #类型转换 将str类型转换成class 'lxml.etree._Element'
#print(type(e))
names=e.xpath('//div[@class="book-mid-info"]/h4/a/text()')
authors=e.xpath('//p[@class="author"]/a[1]/text()')
#print(names)
#print(authors)
for name,author in zip(names,authors):
    print(name,":",author)


