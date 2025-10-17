# AI大模型工程师 简历指导

## 1、大模型的项目，不能超过2个或者3个

```
Java+ AI大模型 ： 1个大模型的项目经验
AI大模型应用工程师： 2个 
AI算法开发工程师： 3个
```

## 专业技能

### 大模型算法方面

1. 熟练使用数据分析可视化三剑客 NumPy、Pandas、Matplotlib 进行数据处理和数据可视化。能熟练掌握Python编程，具备良好的数据分析和问题解决能力。
2. 理解主流机器学习算法原理：线性回归、逻辑回归、Softmax回归，SVM、K-Means、决策树、XGBoost，GBDT 等。
3. 理解主流深度学习算法原理：熟练使用PyTorch框架，熟悉CNN 、RNN、LSTM、GRU、Seq2Seq、Attention、迁移学习、transformer、Bert以及word2vec语言模型。
4. 熟练使用已有机器学习算法进行模型训练和预测，有炼丹经验，参加过阿里天池等比赛项目。
5. 理解 MoE 、Dense、注意力机制、Embedding、Transformer、BERT、GPT 等模型原理和架构。
6. 熟悉大模型的高效参数微调:p-tuning、prefx tuning、lora等微调技术，熟悉大模型的模型压缩方法
7. 熟悉主流的 CV 算法：YOLOv5、YOLOv8，对小目标检测、缺陷检测改进经验。
8. 熟悉 深度学习框架 Pytorch 的网络架构模型，熟练使用里面的 API 进行模型搭建。
9. 掌握 AI 大模型项目开发流程：项目需求分析->数据集制作->模型微调训练->评估测试部署。
10. 熟练使用远程服务器对 LLaMa3 进行 Lora 微调测试评估、模型合并与量化、部署量化后的模型。

### 大模型应用开发方面

1. 熟悉Langchain开发框架，掌握 LLM, Chat, models, PromptTemplates, OutputParse, Chains 等组件的使用。
2. 能够使用SpringAI开发MCP的服务端，掌握SSE和Streamable的通信机制，并结合Agent进行异步开发。
3. 能够自定义 FunctionTool，实现工具的同异步调用。实现聊天、天气查询和网页搜索的 Agent 调用功能。
4. 能够基于 bge-large-zh-v1.5 私有化实现向量数据库Similarity Search、RAG 增强检索。
5. 熟练使用UnstructuredLoader 处理多种格式（JSON、PDF、HTML、Markdown、CSV）包括结构化提取和OCR识别 图文提取。
6. 熟练使用Chroma 、Milvus等向量数据库的分布式部署，构建本地知识库，实现存储与查询。
7. 掌握Dense向量 ，Sparse向量的混合检索，并基于结果进行ReRanker模型的重排。
8. 掌握 LangGraph 核心组件，理解 WorkFlow的构建 和 React Agent 实现思路。 实现过高阶 Agent
   用法 和 多Agent的工作流定制。
9. 熟悉RAG + RAGAS 结合WorkFlow实现动态路由和多 知识库的检索 和 评估。
10. 熟悉Dify + Ollama + Coze + Cursor等各种大模型应用工具的使用。能够基于Dify完成RAG和工作流的定制化。
11. 熟悉基于vllm的各种大模型私有化部署，包括但不限于：DeepSeek，Qwen2.5，Qwen3，Llama-3.2等。

### HuggingFace和模型预训练

1. 熟练使用 Hugging Face API 调用模型，实现文本生成、文本分类和文本问答等 NLP 任务。
2. 熟练使用 datasets API 下载数据集，并对句子进行批量编码。
3. 熟练使用本地模型对下游任务做针对性的模型设计，并训练评估和测试。
4. 掌握模型微调的三种基本模式（全量/增量/局部），使用 BERT 微调过中文评价情感分析数据集。
5. 掌握通过更改模型配置信息并初始化模型，来实现超长文本训练方法，实现文本多分类任务。
6. 熟练使用 ModelScope 在线训练平台进行数据下载和模型调用，在线微调过 Qwen-2.5模型。
7. 熟悉自己租用算力服务器，实现MoE架构的大模型预训练过程。

## 项目经验

### 携程AI智能助手

**项目背景:**
随着大语言模型的发展，传统的智能问答系统已难以满足复杂场景下的多轮对话和动态任务执行需求。为此，本项目基于 LangChain + LangGraph 框架开发多个Agent + WorkFlow实现了一套智能服务助手，聚焦于多轮对话管理、多任务链路执行和工具调用等能力。

**项目描述：**

围绕旅游/出行业务场景，结合携程式服务流程，数据涵盖航班信息、酒店信息、景点推荐等核心领域，替代客服和销售人员，通过交互的方式，完成携程旅游系统中各个业务。由AI大模型自主完成：机票预订，改签，酒店查询和订单管理，旅游产品销售和预定等

**项目实现：**

主要包括业务工具模块、身份信息和权限控制模块、多Agent调度块、状态管理模块。

1、工具函数模块：将携程已有的业务，提取整合成为LangChain工具;依据功能风险将工具划分为“只读工具”与“敏感操作工具”，将所有工具安装业务分类（机票业务，酒店业务，旅游业务，租车业务，公共业务），并与大模型完成交互绑定，使大模型能够动态触发工具调用。

2、身份信息和权限控制模块：主助手智能体（Agent）一开始就能通过节点fetch_user_info从请求参数中获取旅客ID，并调用Tools查询完整的旅客信息，并保存在State中，当整个工作流执行到任何敏感工具节点之前都会进行interrupt_before（中断），并且通过MemorySaver()保存上下文信息，即使中断的情况下Agent也能通过Confg中的thread_id和旅客ID来保证上下文统一。

3、多Agent调度模块： 整个模块主要是把业务流程细分为四个子工作流，由一个主Agent进行居中调度，每个子工作流作为一个单独的Agent负责完成各自的业务，分为：机票业务Agent、酒店业务Agent、旅游业务Agent，租车业务Agent。每个（Agent）子工作流包含五个节点，分别是入口节点(create_entry_node)、决策节点、只读工具、敏感操作工具和离开当前任务节点，从主工作流到子工作流的调度是通过动态路由条件(add_conditional_edges)来实现，从子工作流到主工作流是通过控制State中的update_dialog_stack来管理记录
4、状态管理模块:整个模块是State模块，就是保存工作流中的messages（历史消息列表）、用户信息(user_info)、以及工作流状态栈(dialog_stack)。

**遇到的问题：**

工作流执行过程中，需要在各个子Agent进行切换，会出现上下文语义中断的问题，大模型有概率无法识别语义的切换从而找不到合适的Tools。

如何解决的（可以不写）：给每个子Agent定义对应的输出解析器，并通过大模型从上一个Agent的消息列表中收集当前Agent所需参数，通过闭包给每个子Agent定制动态入口节点和一个离开节点，在入口节点中创建当前子Agent的状态，压入到状态栈(dialog_stack)的顶部，在离开节点中，删除状态栈(dialog_stack)的栈顶状态。

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/482/1748077892057/7ba4cf3183844917b27557d8295ed88c.jpg)

### 基于Deepseek-V3模型RAG的半导体文档知识库系统

**项目背景:**
针对半导体领域大模型知识滞后、幻觉率高的问题（传统方案问答准确率仅62%），构建一个结合技术论文/研报的RAG系统，实现实时知识更新与可验证回答，支撑内部投研人员高效获取精准信息。**技术栈**：LangChain、LangGraph、Milvus、UnstructuredLoader、PyMuPDF、DeepSeek-V3、RAGAS。

**项目实现：**

1、向量数据库在用弹性、分布式部署的Milvus数据库集群，采用Docker部署，可弹性增加节点。

2、使用PyMuPDF解析PDF/PPT文档。通过Linux服务器的私有化部署UnstructuredLoader。使用UnstructuredLoader加载非结构化文档（如 PDF、Word、HTML 等）。设置 hi_res策略来提取章节层级、表格等。通过**Tesseract** （OCR）提取图片信息。 采用章节段落分块 +SemanticChunker语义分块相结合的方式来解决表格或者内容跨页断裂问题。

3、采用私有化部署的bge-large-zh-v1.5和BM25对Chunk进行密集向量化。创建密集向量Dense和稀疏向量Sparse的索引。其中稀疏向量索引采用DAAT算法，控制词频饱和度参数BM25_k1，加速查询性能。稠密向量索引设置构建图时每个节点的最大邻接数M为30，efConstruction=64，提高搜索广度和精度。实现"向量+关键词"混合检索。

4、通过LangGraph定制工作流，实现动态路由能力，根据上下文语音去不同的知识库或者网络进行检索。

5、引入Corrective RAG和Adaptive RAG构建两层评估体系。第一层：从知识库中检索出来的doc进行相关性评估。评估不达标通过Transform_Query触发重新检索。第二层：在输出答案之前进行输入的关联性评估。全部评估通过最终输出答案。

**遇到的问题：**

在线UnstructuredLoader速度慢的问题。 从HuggingFace

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/482/1748077892057/f071de51c0ea4665ab9f6ad6a5e9786b.png)