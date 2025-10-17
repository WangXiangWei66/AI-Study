#教育机构 ：马士兵教育
#讲    师：杨淑娟

import  urllib.request
import  urllib.error

url='https://movie.douban.com/'
try:
    resp=urllib.request.urlopen(url)
except urllib.error.HTTPError as e:
    print('原因:',e.reason)
    print('响应状态码:',str(e.code))
    print('响应头数据:',e.headers)