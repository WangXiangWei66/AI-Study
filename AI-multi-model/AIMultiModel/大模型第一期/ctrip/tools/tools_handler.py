'''本代码实现了LangChain框架中工具调用的错误处理机制、带回退功能的工具节点创建，以及对话时间的打印功能'''


from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables.base import RunnableLambda
from langgraph.prebuilt import ToolNode

#state是一个字典，包含了当前对话的状态信息，包括错误信息和消息列表。
def handle_tool_error(state) -> dict:
    """
       处理工具调用时发生的错误。

       参数:
           state (dict): 当前的状态字典，包含错误信息和消息列表。

       返回:
           dict: 包含错误信息的新消息列表的字典。
       """

    error = state.get("error")  # 获取错误信息
    tool_calls = state["messages"][-1].tool_calls  # 获取最后一条消息中的所有工具调用，用于关联错误到具体的工具调用
    #使用列表推导式为每一个工具调用生成一个ToolMessage
    return {
        "messages": [
            ToolMessage(
                content=f"错误:{repr(error)}\n请修正您的错误。",
                tool_call_id=tc["id"],  # 关联到发生错误的工具调用ID
            )
            for tc in tool_calls  # 遍历所有的工具调用并生成对应的消息
        ]
    }

#带回退机制的工具节点创建函数
#接受一个工具列表，创建ToolNode实例，用于管理执行工具列表中的工具
#为工具节点添加回退处理逻辑，当工具执行抛出异常时触发
def create_tool_node_with_fallback(tools: list) -> dict:
    """
       创建一个带有回退机制的工具节点。当指定的工具执行失败时（例如抛出异常），将触发回退操作。

       参数:
           tools (list): 工具列表。

       返回:
           dict: 带有回退机制的工具节点。
       """
    return ToolNode(tools).with_fallbacks(  #用于添加回退处理
        # 使用handle_tool_error函数作为火腿处理
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

#事件打印函数
def _print_event(event: dict, _printed: set, max_length=1500):
    """
        打印事件信息，特别是对话状态和消息内容。如果消息内容过长，会进行截断处理以保证输出的可读性。

        参数:
            event (dict): 事件字典，包含对话状态和消息。
            _printed (set): 已打印消息的集合，用于避免重复打印。
            max_length (int): 消息的最大长度，超过此长度将被截断。默认值为1500。
        """
    current_state = event.get("dialog_state")  #获取对话状态列表
    if current_state:
        print("当前处于：", current_state[-1])  # 输出当前对话状态
    #消息内容打印
    messages = event.get("messages")
    if messages:
        if isinstance(messages, list):
            message = messages[-1]  # 如果消息是列表，则取最后一个
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)  #获取格式化的消息表示
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + "...(已截断)"  # 超过最大长度则截断
            print(msg_repr)
            _printed.add(message.id)  # 将消息ID添加到已打印集合中
