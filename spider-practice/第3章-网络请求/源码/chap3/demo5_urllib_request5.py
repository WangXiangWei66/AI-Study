#教育机构 ：马士兵教育
#讲    师：杨淑娟
import urllib.request
url='https://www.baidu.com'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}
#构建请求对象
req=urllib.request.Request(url,headers=headers)

#获取opener对象
opener=urllib.request.build_opener()

resp=opener.open(req)
print(resp.read().decode())



