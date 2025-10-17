#教育机构 ：马士兵教育
#讲    师：杨淑娟

import  requests
import  re
url='https://www.qiushibaike.com/video/'
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.100 Safari/537.36'}
resp=requests.get(url,headers=headers)
#print(resp.text)
info=re.findall('<source src="(.*)" type=\'video/mp4\' />',resp.text)
#print(info)
lst=[]
for item in info:
    lst.append('https:'+item)
#print(lst)

count=0
for item in lst:
    count+=1
    resp=requests.get(item,headers=headers)
    with open('video/'+str(count)+'.mp4','wb') as file:
        file.write(resp.content)
print('视频下载完毕')