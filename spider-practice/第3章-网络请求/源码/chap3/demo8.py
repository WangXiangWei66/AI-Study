#教育机构 ：马士兵教育
#讲    师：杨淑娟

import  urllib.request
import  urllib.error

url='http://www.google.com'
#url='http://www.google.cn'  #正确的网址
try:
    resp=urllib.request.urlopen(url)

except urllib.error.URLError as e:
    print(e.reason)