#教育机构 ：马士兵教育
#讲    师：杨淑娟
#发送不带参数的get请求
import  requests
url='http://www.baidu.com'
resp=requests.get(url)
#设置响应的经编码格式
resp.encoding='utf-8'
cookie=resp.cookies #获取请求后的cookie信息
headers=resp.headers
print('响应状态码:',resp.status_code)
print('请求后的cookie:',cookie)
print('获取请求的网址:',resp.url)
print('响应头:',headers)
print('响应内容:',resp.text)



