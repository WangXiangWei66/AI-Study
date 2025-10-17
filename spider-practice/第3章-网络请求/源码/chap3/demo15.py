#教育机构 ：马士兵教育
#讲    师：杨淑娟

import  requests
url='https://www.xslou.com/login.php'
data={'username':'18600605736','password':'57365736','action':'login'}

#使用session发送请求
session=requests.session()
resp=session.post(url,data=data)
resp.encoding='gb2312'

#print(resp.text)
#推荐小说
hot_url='https://www.xslou.com/modules/article/uservote.php?id=71960'
resp2=session.get(hot_url)
resp2.encoding='gb2312'
print(resp2.text)


