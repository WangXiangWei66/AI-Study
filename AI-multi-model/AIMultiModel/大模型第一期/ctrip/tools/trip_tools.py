'''本代码实现了旅行推荐系统的核心功能'''
#代码使用了SQLite数据库存储旅行推荐数据。
from typing import Optional, List  #typing：提供类型注解，如Optional（可选类型）和List（列表类型）。
from sqlite3 import connect, Cursor #sqlite3：用于连接和操作 SQLite 数据库，connect用于创建数据库连接，Cursor用于执行 SQL 查询
from langchain_core.tools import tool
#定义文件路径名
db = "../travel_new.sqlite"


def transform_location(chinese_city):
    # 中文到英文的城市名映射表
    city_dict = {
        '北京': 'Beijing',
        '上海': 'Shanghai',
        '广州': 'Guangzhou',
        '深圳': 'Shenzhen',
        '成都': 'Chengdu',
        '杭州': 'Hangzhou',
        '巴塞尔': 'Basel',
        '苏黎世': 'Zurich',
        # 添加更多的城市映射...
    }

    # Check if the input is in Chinese
    if all('\u4e00' <= char <= '\u9fff' for char in chinese_city):
        return city_dict.get(chinese_city, "城市名称未找到")
    else:
        return chinese_city

#旅行推荐搜索工具
@tool
def search_trip_recommendations(
        location: Optional[str] = None,
        name: Optional[str] = None,
        keywords: Optional[str] = None, #旅行推荐的关联词
) -> List[dict]:
    """
        根据位置、名称和关键词搜索旅行推荐。

        参数:
            location (Optional[str]): 旅行推荐的位置。默认为None。
            name (Optional[str]): 旅行推荐的名称。默认为None。
            keywords (Optional[str]): 关联到旅行推荐的关键词。默认为None。

        返回:
            list[dict]: 包含匹配搜索条件的旅行推荐字典列表。
        """
    conn = connect(db)
    cursor = conn.cursor()  #创建游标
    location = transform_location(location)
    query = "SELECT * FROM trip_recommendations WHERE 1=1" #保证条件始终为真，便于后序拼接
    params = []
    #下面的参数均为模糊匹配
    if location:
        query += "AND location LIKE ?"
        params.append(f"%{location}%")

    if name:
        query += "AND name LIKE ?"
        params.append(f"%{name}%")

    if keywords:
        keyword_list = keywords.split(',')
        keyword_conditions = 'OR'.join(["keywords LIKE ?" for _ in keyword_list])
        query += f"AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    cursor.execute(query, params)
    results = cursor.fetchall() #获取所有结果

    conn.close()
    return [
        dict(zip([column[0] for column in cursor.description], row)) for row in results
    ]

#旅行推荐预定工具
@tool
def book_excursion(recommendation_id: int) -> str:
    """
       通过推荐ID预订一次旅行项目。

       参数:
           recommendation_id (int): 要预订的旅行推荐的ID。

       返回:
           str: 表明旅行推荐是否成功预订的消息。
       """
    conn = connect(db)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE trip_recommendations SET booked = 1 WHERE id = ?",
        (recommendation_id,)
    )
    conn.commit() #提交事务
    #检查受影响的行数
    if cursor.rowcount > 0:
        conn.close()
        return f"旅行推荐{recommendation_id}成功预定。"
    else:
        conn.close()
        return f"未找到与ID相关的旅行推荐信息。{recommendation_id}."

#旅行推荐更新工具
@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
        根据ID更新旅行推荐的详细信息。

        参数:
            recommendation_id (int): 要更新的旅行推荐的ID。
            details (str): 旅行推荐的新详细信息。

        返回:
            str: 表明旅行推荐是否成功更新的消息。
        """
    conn = connect(db)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE trip_recommendations SET details = ? WHERE id = ?",
        (details, recommendation_id)
    )
    conn.commit()
    if cursor.rowcount > 0:
        conn.close()
        return f"旅行推荐{recommendation_id}成功更新。"
    else:
        conn.close()
        return f"未找到{recommendation_id}的旅行推荐。"

#旅行推荐取消工具
@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
        根据ID取消旅行推荐。

        参数:
            recommendation_id (int): 要取消的旅行推荐的ID。

        返回:
            str: 表明旅行推荐是否成功取消的消息。
        """
    conn = connect(db)
    cursor = conn.cursor()
    # 将booked字段设置为0来表示取消预定
    cursor.execute(
        "UPDATE trip_recommendations SET booked = 0 WHERE id = ?",
        (recommendation_id,)
    )
    conn.commit()

    if cursor.rowcount > 0:
        conn.close()
        return f"旅行推荐{recommendation_id}成功取消。"
    else:
        conn.close()
        return f"未找到ID为{recommendation_id}的旅行推荐。"
