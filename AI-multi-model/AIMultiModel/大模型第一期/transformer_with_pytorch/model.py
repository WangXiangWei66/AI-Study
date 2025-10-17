# 本代码实现了Transformer模型的完整结构，是机器翻译等序列转化任务的核心模型
import math  # 导入数学工具库，用于计算 positional encoding 等。
# 导入PyTorch及神经网络模块，用于构建模型
import torch
from torch import nn


# 输入嵌入层
class InputEmbeddings(nn.Module):
    # 将输入的token ID序列转化为高维稠密向量，并通过缩放保证向量级稳定
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # 嵌入维度（如512）
        self.vocab_size = vocab_size  # 词汇表大小
        self.embeddings = nn.Embedding(vocab_size, d_model)  # 嵌入层：将token ID转为d_model维向量

    def forward(self, x):
        # 嵌入后乘以sqrt(d_model)，是Transformer的标准操作，用于平衡嵌入向量的量级
        return self.embeddings(x) * math.sqrt(self.d_model)


# 位置编码层
class PositionalEncoding(nn.Module):
    # 为输入序列添加位置信息（Transformer 本身是无位置感知的），通过正弦 / 余弦函数生成位置编码，使模型能区分不同位置的 token。
    # 核心思想：不同频率的正弦、余弦函数能表示位置的相对关系
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # 嵌入维度
        self.seq_len = seq_len  # 序列最大长度
        self.dropout = nn.Dropout(dropout)  # Dropout层，防止过拟合
        # 初始化位置编码矩阵：(seq_len, d_model)，存储每个位置的编码
        pe = torch.zeros(seq_len, d_model)
        # 生成位置索引：(seq_len, 1)，如[0,1,2,...,seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # 计算衰减因子：用于生成正弦/余弦函数的频率，公式来自Transformer论文
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数位置用正弦编码，奇数位置用余弦编码（交替设置）
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2表示步长为2，取偶数索引
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2表示步长为2，取奇数索引
        # 增加批次维度：(1, seq_len, d_model)，适配批量输入
        pe = pe.unsqueeze(0)
        # 将位置编码注册为非训练参数（缓冲区），不参与反向传播
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输入x与位置编码相加（仅取x的序列长度部分，避免超长）
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # requires_grad_(False)确保不更新
        return self.dropout(x)  # 应用Dropout


# 层归一化
class LayerNormalization(nn.Module):
    # 稳定网络训练，通过归一化层输入的均值和方差，加速收敛。
    # 与 BatchNorm 的区别：LayerNorm 在单样本内计算统计量，BatchNorm 在批次内计算，更适合序列数据。
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 防止除零的小值
        # 可学习的缩放参数（alpha）和偏移参数（bias），初始化为1和0
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # 计算最后一个维度的均值和标准差（保持维度，便于广播）
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 归一化公式：(x - mean) / (std + eps)，再通过alpha和bias调整
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# 前馈网络
class FeedForwardBlock(nn.Module):
    # 对每个 token 进行独立的非线性变换，增强模型的表达能力。
    # 通过升维，增加模型容量，再降维回原维度
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # 第一层线性变换：升维到d_ff（如2048）
        self.dropout = nn.Dropout(p=dropout)  # Dropout层
        self.linear_2 = nn.Linear(d_ff, d_model)  # 第二层线性变换：降维回d_model（如512）

    def forward(self, x):
        # 流程：线性变换→ReLU激活→Dropout→线性变换
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# 多头注意力
class MultiHeadAttentionBlock(nn.Module):
    # 通过多头并行计算注意力，捕捉冈子空间的依赖关系（如语法、语义）
    # 核心流程：线性投影→拆分多头→计算注意力→拼接→输出投影。
    # 掩码作用：在解码时掩盖未来 token（因果掩码），或掩盖填充 token（Padding 掩码）。
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()

        self.d_model = d_model  # 模型维度
        self.h = h  # 头数（如8）
        assert d_model % h == 0, "d_model必须可以整除h"  # 确保每个头的维度是整数
        self.d_k = d_model // h  # 每个头的维度（如512/8=64）
        # 4个线性层：分别用于Q（查询）、K（键）、V（值）的投影，以及多头结果的拼接
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Q的权重矩阵
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # K的权重矩阵
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # V的权重矩阵
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # 输出的权重矩阵

        self.dropout = nn.Dropout(dropout)  # Dropout层

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]  # 每个头的维度（如64）
        # 计算注意力分数：(Q·K^T) / sqrt(d_k)，缩放避免数值过大
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # 应用掩码：将掩码为0的位置设为-1e9（softmax后接近0，不关注）
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        # 对分数做softmax，得到注意力权重（总和为1）
        attention_scores = attention_scores.softmax(dim=-1)
        # 应用Dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # 注意力输出：权重·V，同时返回注意力分数（可选，用于可视化）
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # 1. 对Q、K、V进行线性投影
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)
        # 2. 拆分多头：将d_model拆分为h个d_k
        # 形状变化：(batch_size, seq_len, d_model) → (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # 3. 计算多头注意力
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # 4. 拼接多头结果：(batch_size, h, seq_len, d_k) → (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # 5. 输出投影
        return self.w_o(x)


# 残差连接
class ResidualConnection(nn.Module):
    # 缓解深层网络的梯度消失问题，使模型更容易训练
    # 设计细节：Transformer 采用 “先归一化后子层” 的结构（与原论文相反，但实践中更稳定）。
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.norm = LayerNormalization(features=features)  # 层归一化

    def forward(self, x, sublayer):
        # 残差连接公式：x + Dropout(sublayer(LayerNorm(x)))
        # 先归一化，再经过子层（如注意力或前馈网络），再Dropout，最后与原x相加
        return x + self.dropout(sublayer(self.norm(x)))


# 编码器块
import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """残差连接模块（用于自注意力和前馈网络后）"""

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(features)  # 层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout防止过拟合

    def forward(self, x, sublayer):
        # 残差连接公式：x + dropout(sublayer(norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """编码器块：由自注意力模块 + 前馈网络模块组成，带残差连接"""

    def __init__(self,
                 features: int,  # 特征维度（与d_model一致）
                 self_attention_block: nn.Module,  # 自注意力模块（如多头注意力）
                 feed_forward_block: nn.Module,  # 前馈网络模块
                 dropout: float):  # Dropout概率
        super().__init__()
        self.self_attention_block = self_attention_block  # 自注意力模块
        self.feed_forward_block = feed_forward_block  # 前馈网络模块
        # 2个残差连接：分别用于自注意力层和前馈网络层
        self.residual_connection = nn.ModuleList([
            ResidualConnection(features, dropout) for _ in range(2)
        ])

    # 关键修复：forward方法必须与__init__同级，不能缩进在__init__内部
    def forward(self, x, src_mask):
        # 1. 自注意力层（Q=K=V=x），应用第一个残差连接
        # 注意：使用lambda封装自注意力模块，传入参数x, x, x, src_mask
        x = self.residual_connection[0](
            x,
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # 2. 前馈网络层，应用第二个残差连接
        # 前馈网络只需输入x，无需额外参数
        x = self.residual_connection[1](
            x,
            self.feed_forward_block
        )

        return x

# 编码器
class Encoder(nn.Module):
    # 作用：将输入序列编码为上下文感知的向量表示，传递给解码器。
    # 堆叠设计：多层编码器块堆叠可捕捉更复杂的上下文信息（如长距离依赖）。
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers  # 多个编码器块的列表（如6个）
        self.norm = LayerNormalization(features=features)  # 最终的层归一化

    def forward(self, x, mask):
        # 依次通过每个编码器块
        for layer in self.layers:
            x = layer(x, mask)
            # 输出前再做一次归一化
        return self.norm(x)


# 解码器块
class DecoderBlock(nn.Module):
    # 作用：解码器的基本单元，由“掩码自注意力 + 交叉注意力+前馈网络”组成
    # 掩码子注意力：确保解码时只能看到当前及之前的token（如翻译时不能提前看到后面的词）
    # 交叉注意力：使解码器关注编码器输出的相关信息（如源语言的关键内容）
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block  # 解码器自注意力（掩码自注意力）
        self.cross_attention_block = cross_attention_block  # 交叉注意力（与编码器输出交互）
        self.feed_forward_block = feed_forward_block  # 前馈网络
        # 3个残差连接：分别用于自注意力、交叉注意力、前馈网络
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

        def forward(self, x, encoder_output, src_mask, tgt_mask):
            # 1. 解码器自注意力：Q=K=V=x，输入目标掩码（掩盖填充和未来token）
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
            # 2. 交叉注意力：Q=x（解码器输出），K=V=encoder_output（编码器输出），输入源掩码
            x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                    src_mask))
            # 前馈网络层
            x = self.residual_connection[2](x, self.feed_forward_block)
            return x


# 解码器
class Decoder(nn.Module):
    # 根据编码器输出和已生成的目标序列，生成下一个token的分布
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers  # 多个解码器块的列表（如6个）
        self.norm = LayerNormalization(features=features)  # 最终的层归一化

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 依次通过每个解码器块
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            # 输出前再做一次归一化
        return self.norm(x)


# 投影层
class ProjectionLayer(nn.Module):
    # 将解码器的输出向量转换为词汇表中每个 token 的得分，后续通过 softmax 得到概率。
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # 线性投影：d_model→词汇表大小

    def forward(self, x):
        # 将解码器输出投影到词汇表空间，用于计算每个token的概率
        return self.proj(x)  # 输出形状：(batch_size, seq_len, vocab_size)


# Transformer主模型
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: PositionalEncoding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 源语言嵌入
        self.tgt_embed = tgt_embed  # 目标语言嵌入
        self.src_pos = src_pos  # 源语言位置编码
        self.tgt_pos = tgt_pos  # 目标语言位置编码
        self.projection_layer = projection_layer  # 投影层

    # 源语言序列编码流程：嵌入→位置编码→编码器
    def encode(self, src, src_mask):
        src = self.src_embed(src)  # 源语言 token 嵌入
        src = self.src_pos(src)  # 加入位置编码
        return self.encoder(src, src_mask)  # 编码器处理

    # 目标语言序列解码流程：嵌入→位置编码→解码器
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)  # 目标语言 token 嵌入
        tgt = self.tgt_pos(tgt)  # 加入位置编码
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # 解码器处理

    # 投影到词汇表空间
    def project(self, x):
        return self.projection_layer(x)


# Transformer构建函数
def builder_transformers(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                         d_model: int = 512,
                         N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # 1. 初始化嵌入层和位置编码
    src_embed = InputEmbeddings(d_model, src_vocab_size)  # 源语言嵌入
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)  # 目标语言嵌入

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)  # 源语言位置编码
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)  # 目标语言位置编码
    # 构建编码器块
    encoder_blocks = []
    for _ in range(N):  # N为编码器块数量（如6）
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 编码器自注意力
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # 前馈网络
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)  # 编码器块
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))  # 编码器
    # 构建解码器
    decoder_blocks = []
    for _ in range(N):  # N为解码器块数量（如6）
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 解码器自注意力
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)  # 交叉注意力
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)  # 前馈网络
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)  # 解码器块
        decoder_blocks.append(decoder_block)

    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))  # 解码器
    # 4. 投影层（目标是输出目标语言词汇表
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # 5. 构建Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    # 6. 参数初始化：对所有可学习参数（维度>1）使用Xavier均匀初始化
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
