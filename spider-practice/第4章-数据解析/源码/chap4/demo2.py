#教育机构 ：马士兵教育
#讲    师：杨淑娟

from  bs4 import  BeautifulSoup

html='''
    <html>
        <head>
            <title>马士兵教育</title>
        </head>
        <body>
            <h1 class="info bg" float='left'>欢迎大家来到马士兵教育</h1>
            <a href="http://www.mashibing.com"> 马士兵教育</a>
            <h2><!--注释的内容--></h2>
        </body>
    </html>

'''
#bs=BeautifulSoup(html,'html.parser')
bs=BeautifulSoup(html,'lxml')
print(bs.title)   #获取标签
print(bs.h1.attrs)  #获取h1标签的所有属性

#获取单个属性
print(bs.h1.get('class'))
print(bs.h1['class'])
print(bs.a['href'])

#获取文本内容
print(bs.title.text)
print(bs.title.string)

#获取内容
print('--------',bs.h2.string)  #获取到h2标签中的注释的文本内容
print(bs.h2.text)     #因为h2标签中没有正而八经的文本内容
