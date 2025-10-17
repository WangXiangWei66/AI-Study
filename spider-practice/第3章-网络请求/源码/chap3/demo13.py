#教育机构 ：马士兵教育
#讲    师：杨淑娟
import  requests
url='https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png'
resp=requests.get(url)

#存储
with open('logo.png','wb')  as file:
    file.write(resp.content)

