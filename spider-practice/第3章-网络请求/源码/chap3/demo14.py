#教育机构 ：马士兵教育
#讲    师：杨淑娟

import  requests
url='https://www.xslou.com/login.php'
data={'username':'18600605736','password':'57365736','action':'login'}
resp=requests.post(url,data=data)
resp.encoding='gb2312'
print(resp.status_code)
print(resp.text)
