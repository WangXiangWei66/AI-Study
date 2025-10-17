#教育机构 ：马士兵教育
#讲    师：杨淑娟
#开发时间：2020/5/7 12:16
my_json={'name':'Python','address':{'province':'吉林省','city':['长春市','吉林市','松原市']}}

#获取吉林省
province=my_json['address']['province']
print(province)

#获取吉林市
city=my_json['address']['city'][1]
print(city)
