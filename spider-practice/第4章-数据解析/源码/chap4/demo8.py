#教育机构 ：马士兵教育
#讲    师：杨淑娟

from  pyquery import  PyQuery
doc=PyQuery(url='http://www.baidu.com',encoding='utf-8')
print(doc('title'))