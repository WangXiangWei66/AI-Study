# coding : utf-8
#本代码主要定义了一个及其翻译模型的配置参数，以及获取模型权重文件的工具函数
from pathlib import Path #从 pathlib 模块导入 Path 类，用于便捷地处理文件路径

#用于返回模型的配置参数，返回一个字典
def get_config():
    return {
        "batch_size": 8,#每次训练的批次大小，依次处理8个样本
        "num_epoch": 20,#训练的总轮数，整个数据集将被训练20次
        "lr": 10 ** -4, #学习率
        "seq_len": 350,#序列长度，输入模型的文本最大长度为350个字符
        "d_model": 512,#模型中特征向量的维度
        "datasource": "opus_books", #数据源名称，这里使用opus_books数据集
        "lang_src": "en",#源语言为英文
        "lang_tgt": "it", # 目标语言，意大利语
        "model_folder": "weights",# 模型权重文件存放的文件夹名
        "model_basename": "tmodel_",# 模型权重文件的基础名称前缀
        "tokenizer_file": "tokenizer_{0}.json",# 分词器文件的命名格式，{0}会被替换为语言代码
        "experiment_name": "runs/tmodel"# 实验日志存放路径
    }

#用于根据配置和伦茨获取模型权重文件的路径
#config：配置字典
#epoch：训练轮次（字符串类型）
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"#拼接模型文件夹名，格式为 "数据源_权重文件夹名"（例如 "opus_books_weights"）
    model_filename = f"{config['model_basename']}{epoch}.pt"#拼接模型文件名，格式为 "模型前缀 + 轮次.pt"（例如 "tmodel_5.pt" 表示第 5 轮的模型）
    return str(Path('.') / model_folder / model_filename)#使用 Path 构造完整路径（当前目录 / 模型文件夹 / 模型文件名）
