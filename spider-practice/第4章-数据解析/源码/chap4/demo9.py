#教育机构 ：马士兵教育
#讲    师：杨淑娟
from  pyquery import  PyQuery
doc=PyQuery(filename='demo.html')
print(doc)
print(doc('h1'))

