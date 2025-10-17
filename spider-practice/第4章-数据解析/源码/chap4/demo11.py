#教育机构 ：马士兵教育
#讲    师：杨淑娟
import  requests
from  pyquery import  PyQuery
url='https://www.qidian.com/rank/yuepiao'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}
resp=requests.get(url,headers=headers)
#print(resp.text)
#初始化PyQuery对象
doc=PyQuery(resp.text)  #使用字符串初始化方式初始化PyQuery对象
#a_tag=doc('h4 a')
#print(a_tag)
names=[a.text for a in doc('h4 a')]

authors=doc('p.author a')

author_lst=[]
for index in range(len(authors)):
    if index%2==0:
        author_lst.append(authors[index].text)
#print(author_lst)



#print(names)

for name,author in zip(names,author_lst):
    print(name,':',author)

