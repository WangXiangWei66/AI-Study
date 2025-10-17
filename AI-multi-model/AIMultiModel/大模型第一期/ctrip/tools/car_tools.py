'''本代码定义了四个操作汽车租赁信息的工具函数，他们通过SQLite数据库实现了搜索、预定、更新和取消汽车租赁服务的功能
    这些函数使用了LangChain的@toolde
    装饰器，使其可以作为工具集成到更大的AI系统中'''

from datetime import datetime, date
from sqlite3 import connect
from typing import Optional, Union

from langchain_core.tools import tool  # 将函数标记为可悲AI调用的工具
from sqlalchemy.engine import row

from .location_trans import transform_location

db = "../travel_new.sqlite"  # 这是数据库文件名


@tool
def search_car_rentals(
        location: Optional[str] = None,
        name: Optional[str] = None,
        # price_tier: Optional[str] = None,
        # start_date: Optional[Union[datetime, date]] = None,
        # end_date: Optional[Union[datetime, date]] = None,
) -> list[dict]:
    """
        根据位置、名称、价格层级、开始日期和结束日期搜索汽车租赁信息。

        参数:
        - location (Optional[str]): 汽车租赁的位置。默认为None。
        - name (Optional[str]): 汽车租赁公司的名称。默认为None。
        返回:
        - list[dict]: 包含匹配搜索条件的汽车租赁信息的字典列表。
        """

    conn = connect(db)
    cursor = conn.cursor()
    location = transform_location(location)
    query = "SELECT * FROM car_rentals WHERE 1=1"  # 动态查询SQL语句，避免拼接SQL时空条件问题
    params = []

    if location:
        query += " AND location LIKE?"
        params.append(f"%{location}%")

    if name:
        query += " AND name LIKE?"
        params.append(f"%{name}%")  # 进行了模糊搜索

    # 由于实例数据集没有太多的数据，因此不对日期和价格层级进行严格匹配

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn.close()
    # 最后将查询结果转化为字典列表（键为列名，值为行数据）
    return [
        dict(zip([column[0] for column in cursor.description], row))
    ]


# 通过ID预定汽车
@tool
def book_car_rental(rental_id: int) -> str:
    """
       通过ID预订汽车租赁服务。

       参数:
       - rental_id (int): 要预订的汽车租赁服务的ID。

       返回:
       - str: 表明汽车租赁是否成功预订的消息。
       """
    conn = connect(db)
    cursor = conn.cursor()
    # 使用UPDATE语句修改数据库记录
    cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
    conn.commit()
    # 使用cursor.rowcount判断是否成功更新记录
    if cursor.rowcount > 0:
        cursor.close()
        return f"汽车租赁{rental_id}成功预定。"
    else:
        conn.close()
        return f"未找到ID为{rental_id}的汽车租赁服务。"


# 更新汽车租赁信息，支持部分更新
@tool
def update_car_rental(
        # 根据传入的参数，选择性的更新start_date 或end_date
        rental_id: int,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """
    根据ID更新汽车租赁的开始和结束日期。

    参数:
    rental_id (int): 要更新的汽车租赁服务的ID。
    start_date (Optional[Union[datetime, date]]): 汽车租赁的新开始日期。默认为None。
    end_date (Optional[Union[datetime, date]]): 汽车租赁的新结束日期。默认为None。

    返回:
    str: 表明汽车租赁是否成功更新的消息。
    """
    conn = connect(db)
    cursor = conn.cursor()

    if start_date:
        cursor.execute(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date, rental_id,)
        )
    if end_date:
        cursor.execute(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?",
            (end_date, rental_id,)
        )

    conn.commit()  # 保证数据的一致性

    if cursor.rowcount > 0:
        cursor.close()
        return f"汽车租赁{rental_id}成功更新。"
    else:
        conn.close()
        return f"未找到ID为{rental_id}的汽车租赁服务。"


# 取消汽车租赁信息
@tool
def cancel_car_rental(rental_id: int) -> str:
    """
      根据ID取消汽车租赁服务。

      参数:
          rental_id (int): 要取消的汽车租赁服务的ID。

      返回:
          str: 表明汽车租赁是否成功取消的消息。
      """
    conn = connect(db)
    cursor = conn.cursor()

    # 将booked字段设置为0来表示取消预定
    cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?",
                   (rental_id,))
    conn.commit()

    if cursor.rowcount > 0:
        cursor.close()
        return f"汽车租赁{rental_id}成功取消。"
    else:
        conn.close()
        return f"未找到ID为{rental_id}的汽车租赁服务。"
