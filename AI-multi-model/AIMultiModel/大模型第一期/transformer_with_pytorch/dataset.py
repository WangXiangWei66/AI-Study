'''
本代码定义了一个英语双鱼翻译任务的数据集类BilingualDataset,继承自PyTorch的Dataset，主要功能是
将原始文本数据转换为模型可处理的张量格式，并生成训练所需的输入、标签和掩码
'''
from typing import Any  # 导入类型提示，用于指定 __getitem__ 方法的参数类型。
import torch  # 导入PyTorch库，用于张量操作
from torch.utils.data import Dataset  # 自定义数据集需继承此类。


# 定义双语数据集类，继承自DataSet，用于加载和处理双语平行语料
class BilingualDataset(Dataset):
    '''
    ds:原始数据集（包含双语翻译时的列表）
    tokenizer_src：源语言分词器（将源语言文本转为 token 序列）。
    tokenizer_tgt：目标语言分词器。
    src_lang：源语言代码（如 "en" 表示英语）。
    tgt_lang：目标语言代码（如 "it" 表示意大利语）。
    seq_len：固定序列长度，所有样本将被填充或截断到此长度。
    '''

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()  # 调用父类 Dataset 的初始化方法
        # 将参数保存为实例变量，方便后续方法调用
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        '''
        定义特殊 token 的张量：
        [SOS]：句子开始标记（Start of Sentence）。
        [EOS]：句子结束标记（End of Sentence）。
        [PAD]：填充标记（用于补全序列长度）。
        '''
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    # 数据集长度方法
    def __len__(self):
        return len(self.ds)  # 返回数据集样本的数量（与原始数据集长度一致）

    # 获取样本方法
    def __getitem__(self, index: Any):  # 根据索引index获取单个样本
        src_target_pair = self.ds[index]  # 从原始数据集获取第 index 个样本（包含双语翻译对）。
        src_text = src_target_pair['translation'][self.src_lang]  # 提取源语言文本
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # 提取目标语言文本

        # 有了原始的句子，我们使用分词器进行分词，转化为token对应的ID序列
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 需要填充的token数量
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 编码器输入需要包含 [SOS] 和 [EOS]，因此减去 2。
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 解码器输入只需包含 [SOS]（[EOS] 放在标签中），因此减去 1。
        # 如果原始文本过长（加上特殊标记后超过 seq_len），抛出错误。
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('句子太长，超过了seq_len,输入不全会翻译的不准')

        # 构造真正训练需要的encoder输入和decoder输入、输出
        # 结构：[SOS] + 源语言token ID + [EOS] + [PAD]*(填充数量)。
        # 用 torch.cat 拼接成一个长度为 seq_len 的张量。
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        # 构造标签张量（用于计算损失）
        # 长度为 seq_len（与解码器输入错位一位，实现自回归训练）。
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        # 断言检查：确保编码器输入、解码器输入、标签的长度均为 seq_len，避免格式错误。
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        # 返回一个字典，包含模型训练所需的所有信息
        return {
            'encoder_input': encoder_input,  # 编码器和解码器的输入张量。
            'decoder_input': decoder_input,
            'label': label,  # 训练标签张量。
            'src_text': src_text,  # ：原始文本（用于后续可视化或评估）。
            'tgt_text': tgt_text,
            # encoder_mask：编码器掩码（掩盖 [PAD] 位置，避免模型关注填充 token）
            # (encoder_input != self.pad_token)：生成布尔掩码（True 表示有效 token，False 表示填充）。
            # unsqueeze(0).unsqueeze(0)：增加两个维度，适配模型输入格式。
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # decoder_mask：解码器掩码（既掩盖填充，又掩盖未来 token，实现因果语言模型）。
            # (decoder_input != self.pad_token).unsqueeze(0).int()：填充掩码。
            # causal_mask(...)：因果掩码（下三角矩阵，确保解码器只能看到当前及之前的 token）。
            # &：将填充掩码和因果掩码合并。
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
        }


# 因果掩码函数
'''
torch.ones(1, size, size)：创建一个 1×size×size 的全 1 矩阵。
torch.triu(..., diagonal=1)：取上三角部分（对角线以上为 1，以下为 0）。
mask == 0：转换为布尔掩码（下三角及对角线为 True，上三角为 False），确保解码器只能关注当前及之前的 token，无法看到未来信息。
'''

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
