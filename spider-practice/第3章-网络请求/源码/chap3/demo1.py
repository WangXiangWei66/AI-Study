#教育机构 ：马士兵教育
#讲    师：杨淑娟
#urllib.parse  用于解析url
import urllib.parse
kw={'wd':'马士兵'}
#编码
result=urllib.parse.urlencode(kw)
print(result)

#解码
res=urllib.parse.unquote(result)
print(res)

