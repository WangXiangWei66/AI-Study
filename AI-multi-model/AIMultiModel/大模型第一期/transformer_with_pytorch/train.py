# ds:数据集
# lang:语言标识
# 生成器的作用是高效地逐个返回数据，而不是一次性加载所有数据到内存，适合处理大型数据集
import warnings
from pathlib import Path
import torch
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from model import builder_transformers
from config import get_config, get_weights_file_path
from dataset import BilingualDataset
from datasets import load_dataset
from tokenizers import Tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


# config:配置字典
def get_or_build_tokenizer(config, ds, lang):
    # 从配置中获取分词器文件路径模板，用lang填充模板得到具体路径
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # 如果文件不存在，则创建一个新的分词器，使用BPE算法
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()  # 设置预分词为空格分词
        # 指定特殊 token：未知词、填充符、句子开始符、句子结束符
        trainer = BpeTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        # 使用前面定义的生成器遍历数据集，训练分词器
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))  # 将训练好的分词器保存在指定的路径
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))  # 如果分词器已经存在，则从文件加载分词器
    return tokenizer  # 返回加载或者预训练好的分词器


# 获取并预处理数据集，为机器翻译模型的训练和验证做准备
def get_ds(config):
    '''
    使用 load_dataset 函数加载原始数据集
    config['datasource']：指定数据源（如数据集名称）
    config['lang_src']}-{config['lang_tgt']：指定语言对（如 "en-fr" 表示英语到法语）
    split='train'：只加载训练集部分
    最终将原始的数据集存储在ds_raw变量
    '''
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # 创建tokenizers
    '''
    使用之前定义的函数，为源语言和目标语言分别创建或加载分词器
    '''
    tokenizers_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizers_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90%用于训练  10%用于验证
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    '''
    random_split:用于随机分割数据集
    '''
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    train_ds = BilingualDataset(train_ds_raw, tokenizers_src, tokenizers_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizers_src, tokenizers_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # 计算句子长度的代码
    max_len_src = 0
    max_len_tgt = 0
    # 遍历所有的数据项，对每个句子进行编码，获取其token ID 列表并计算长度
    for item in ds_raw:
        src_ids = tokenizers_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizers_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    # 创建数据加载器，用于 批量加载数据
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    # 最后返回训练数据加载器，验证数据加载器，源语言分词器和目标语言分词器
    return train_dataloader, val_dataloader, tokenizers_src, tokenizers_tgt


# 下面的两个是整个Transformer模型训练流程的核心，负责模型创建、训练循环、损失计算、参数优化等关键工作
def get_model(config, vocab_src_len, vocab_tgt_len):#包含了源语言词汇大小和目标语言的词汇大小
    model = builder_transformers(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])#源 / 目标语言词汇表大小、序列长度（seq_len）、模型隐藏层维度（d_model）
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    if device == 'cuda':
        print(f"Device name:{torch.cuda.get_device_name(device.index)}")
        print(f'Device memory:{torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3:.2f} GB')

    # 保证保存模型weights的文件夹是存在的
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizers_src, tokenizers_tgt = get_ds(config)
    model = get_model(config, tokenizers_src.get_vocab_size(), tokenizers_tgt.get_vocab_size()).to(device).float()

    # 注释掉TensorBoard相关代码
    # writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizers_src.token_to_id('[PAD]'), label_smoothing=0.1)
    initial_epoch = 0
    global_step = 0

    for epoch in range(initial_epoch, config['num_epoch']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch :02d}')

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device).float()
            decoder_input = batch['decoder_input'].to(device).float()
            label = batch['label'].to(device).long()
            encoder_mask = batch['encoder_mask'].to(device).float()
            decoder_mask = batch['decoder_mask'].to(device).float()

            encoder_output = model.encoder(encoder_input, encoder_mask)
            decoder_output = model.decoder(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizers_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # 注释掉TensorBoard日志记录
            # writer.add_scalar('train loss', loss.item(), global_step=global_step)
            # writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    print(config)
    train_model(config)


# def train_model(config):
#     #自动选择训练设备：优先使用 GPU（CUDA），若无则使用 CPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device {device}')#将当前使用的设备进行打印
#     #如果使用的是GPU，额外打印GPU名称和总内存
#     if device == 'cuda':
#         print(f'Device name: {torch.cuda.get_device_name(device.index)}')
#         print(f'Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3}')
#     device = torch.device(device)#确保设备对象正确初始化（冗余代码，可省略）
#
#     # 保证保存模型weights的文件夹是存在的
#     Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
#     '''
#     调用 get_ds 函数（之前解析过）获取：
#     训练数据加载器（train_dataloader）
#     验证数据加载器（val_dataloader）
#     源语言分词器（tokenizers_src）
#     目标语言分词器（tokenizers_tgt）
#     '''
#     train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
#     #将模型移动到指定的设备
#     model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
#
#     base_dir = "D:\\A-For-Study\\PythonFullStack\\pythonFullStack\\AI-multi-model\\AIMultiModel\\大模型第一期\\transformer_with_pytorch"
#     log_dir = os.path.join(base_dir, "runs", "tmodel")
#     # 强制创建目录
#     os.makedirs(log_dir, exist_ok=True)
#     # 初始化SummaryWriter
#     writer = SummaryWriter(log_dir)#记录训练中的指标
#     #初始化优化器：使用 Adam 优化器，学习率从配置中获取（config['lr']），数值稳定性参数 eps=1e-9
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
#     #初始化损失函数，使用交叉熵损失
#     loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
#
#     initial_epoch = 0#起始轮次
#     global_step = 0#全局步数
#
#     # 分轮次分批次的进行训练
#     for epoch in range(0, config['num_epochs']):
#         model.train()
#         #使用 tqdm 创建进度条，显示当前轮次的训练进度
#         batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
#
#         for batch in batch_iterator:
#             '''
#             从批次中提取数据并移动到指定设备：
#             encoder_input：编码器输入（源语言句子的 token ID）
#             decoder_input：解码器输入（目标语言句子的 token ID，用于自回归预测）
#             label：解码器的目标输出（与 decoder_input 偏移一位，用于计算损失）
#             encoder_mask/decoder_mask：掩码（用于屏蔽填充符和未来 token）
#             '''
#             encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
#             decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
#             label = batch['label'].to(device)  # (B, seq_len)
#             encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
#             decoder_mask = batch['decoder_mask'].to(device)  # # (B, 1, seq_len, seq_len)
#             #将编码器输入和掩码传入模型的编码器，得到编码器输出
#             encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
#             #将编码器输出、编码器掩码、解码器输入和解码器掩码传入模型的解码器，得到解码器输出（此处疑似笔误，第一个参数应为 encoder_output 而非 encoder_input）
#             decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
#                                           decoder_mask)  # (B, seq_len, d_model)
#             #将解码器输出传入投影层（project），得到最终的词汇表维度输出（用于预测下一个 token）
#             proj_output = model.project(decoder_output)
#             '''
#             计算损失：
#             proj_output.view(-1, vocab_size)：将输出展平为二维张量（[batch_size*seq_len, vocab_size]）
#             label.view(-1)：将标签展平为一维张量（[batch_size*seq_len]）
#             交叉熵损失会自动计算每个位置的预测概率与真实标签的损失
#             '''
#             loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
#             #在进度条上显示当前批次的损失值
#             batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
#
#             # 记录到 tensorboard 中
#             writer.add_scalar('train loss', loss.item(), global_step=global_step)
#             writer.flush()
#
#             loss.backward()
#
#             optimizer.step()
#             optimizer.zero_grad()#清空梯度，避免下一个批次的梯度与当前批次累积
#
#             global_step += 1
#
#         # 每个轮次完成之后, 我们保存一下模型
#         model_filename = get_weights_file_path(config, f'{epoch:02d}')
#         #保存模型：包括当前轮次、模型参数、优化器参数和全局步数
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),  # 获取模型参数保存下来
#             'optimizer_state_dict': optimizer.state_dict(),  # 获取优化器相关的状态
#             'global_step': global_step
#         }, model_filename)
#
