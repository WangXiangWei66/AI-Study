import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# llm = ChatOpenAI(
#     temperature = 0,
#     model = 'deepseek-chat',
#     base_url="https://api.deepseek.com",
#     api_key="sk-994985da2eb24831a902b05e1ea97088"
# )

# llm = ChatOpenAI(
#     temperature = 0,
#     model = "GLM-4-0520",
#     base_url="https://open.bigmodel.cn/api/paas/v4/",
#     api_key="e9a3fd4df9544e149918a11f426fb8c6.aT7ti3NevK9Zgusv"
# )

llm = ChatOpenAI(
    temperature=1.0,
    model='gpt-3.5-turbo',
    api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
    base_url="https://xiaoai.plus/v1",
    verbose=False
)
# llm = ChatOpenAI(
#     temperature=0,
#     model='claude-3-7-sonnet-2025219',
#     api_key="sk-ZKECakqu96gnIMx1iBsKkhHNl0qX3VnIDOJdtxPzTcFZdD1u",
#     base_url="https://xiaoai.plus/v1",
#     verbose=False
# )
# 初始化搜索工具，限制结果数量为2
os.environ["TAVILY_API_KEY"] = "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"  # 定义Tavily搜索工具的API密钥
tavily_tool = TavilySearch(max_results=2)
