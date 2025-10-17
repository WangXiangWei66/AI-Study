#教育机构 ：马士兵教育
#讲    师：杨淑娟
import  urllib.request

url='https://movie.douban.com/'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}

#构建请求对象
req=urllib.request.Request(url,headers=headers)

#使用urlopen打开请求
resp=urllib.request.urlopen(req)

#从响应结果中读取数据
html=resp.read().decode('utf-8')
print(html)

