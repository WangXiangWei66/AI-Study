#教育机构 ：马士兵教育
#讲    师：杨淑娟

from  pyquery import  PyQuery as py
html='''
    <html>
        <head>
            <title>PyQuery</title>
        </head>
        <body>
            <h1>PyQuery</h1>
        </body>
    </html>
'''
doc=py(html)  #创建PyQuery的对象，实际上就是在进行一个类型转换，将str类型转成PyQuery类型
print(doc)
print(type(doc))
print(type(html))
print(doc('title'))