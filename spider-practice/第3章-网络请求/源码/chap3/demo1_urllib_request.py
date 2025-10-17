#教育机构 ：马士兵教育
#讲    师：杨淑娟
import urllib.request

url='https://www.lingdianshuwu.com/'

#发送请求
resp=urllib.request.urlopen(url)
html=resp.read().decode('gbk')   #decode将bytes类型转成str类型
print(html)

