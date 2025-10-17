#教育机构 ：马士兵教育
#讲    师：杨淑娟
from  bs4 import  BeautifulSoup

html='''
    <title>马士兵教育</title>
    <div class="info" float="left">欢迎来到马士兵教育</div>
    <div class="info" float="right" id="gb">
        <span>好好学习，天天向上</span>
        <a href="http://www.mashibing.com">官网</a>
    </div>
    <span>人生苦短，你需要Python</span>
'''
bs=BeautifulSoup(html,'lxml')
print(bs.title,type(bs.title))
print(bs.find('div',class_='info'),type(bs.find('div',class_='info')))  #获取第一个满足条件的标签
print('--------------------------------------')
print(bs.find_all('div',class_='info'))  #得到的是一个标签的列表
print('--------------------------------------')
for item in bs.find_all('div',class_='info'):
    print(item,type(item))
print('--------------------------------------')
print(bs.find_all('div',attrs={'float':'right'}))

print('===============CSS选择器=======================')
print(bs.select("#gb"))
print('--------------------------------------')
print(bs.select('.info'))
print('--------------------------------------')
print(bs.select('div>span'))
print('--------------------------------------')
print(bs.select('div.info>span'))

for item in bs.select('div.info>span'):
    print(item.text)