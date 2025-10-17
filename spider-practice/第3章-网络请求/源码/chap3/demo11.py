#教育机构 ：马士兵教育
#讲    师：杨淑娟

import  requests
url='https://www.so.com/s'
params={'q':'python'}
resp=requests.get(url,params=params)
resp.encoding='utf-8'
print(resp.text)