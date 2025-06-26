import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    根均方层归一化（RMSNorm）
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放权重
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        # 均方根：先计算均值，再开方
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # 归一化并缩放
        return x / rms * self.weight

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）实现
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        positions = torch.arange(max_seq_len).unsqueeze(1)
        freqs = positions * inv_freq.unsqueeze(0)
        # 拼接 sin 和 cos 两部分
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        # x: (batch, num_heads, seq_len, head_dim)
        seq_len = x.size(-2)
        emb = self.emb[:seq_len, :]
        sin, cos = emb.chunk(2, dim=-1)
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]
        # 拆分 x 为两部分，按公式旋转
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class MultiHeadAttention(nn.Module):
    """
    LLaMA 多头自注意力，支持缓存
    """
    def __init__(self, dim, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim 必须能被 num_heads 整除'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # 线性变换
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # 旋转位置编码
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_key_value=None):
        # x: (batch, seq_len, dim)
        B, T, C = x.shape
        # 投影并拆分多头，变形为 (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        # 应用 RoPE 旋转位置编码
        q = self.rotary(q)
        k = self.rotary(k)
        # 拼接缓存的 k,v 用于生成
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present = (k, v)
        # 计算注意力分数并缩放
        scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        # 构造因果 mask，只保留下三角
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1,1,T,T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # softmax -> attention 权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # 加权 v
        out = torch.matmul(attn, v)
        # 恢复维度 (B, T, C)
        out = out.permute(0,2,1,3).contiguous().view(B, T, C)
        # 输出线性投影
        return self.out_proj(out), present

class MLP(nn.Module):
    """
    LLaMA 前馈网络，采用 SwiGLU 激活
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = dim * mult
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        # SwiGLU 计算：先 w1(x) 然后 siLU，再与 w2(x) 相乘，最后 w3 投影
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Layer(nn.Module):
    """
    LLaMA 单层解码器（含 RMSNorm + Attention + MLP + 残差）
    """
    def __init__(self, dim, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        # 预归一化
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, past_key_value=None):
        # Attention 子层
        h = self.norm1(x)
        attn_out, present = self.attn(h, past_key_value)
        x = x + attn_out  # 残差连接
        # MLP 子层
        m = self.norm2(x)
        x = x + self.mlp(m)  # 残差连接
        return x, present

class Llama(nn.Module):
    """
    LLaMA 模型结构，仅包含网络定义，不含训练和推理逻辑
    """
    def __init__(self, vocab_size, max_seq_len, dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        # Token 嵌入
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        # 由多层解码器组成
        self.layers = nn.ModuleList([
            Layer(dim, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        # 最终归一化
        self.norm = RMSNorm(dim)
        # 输出层，生成 logits
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None):
        """
        input_ids: (batch, seq_len) 整数 token id
        past_key_values: 上一时刻缓存，用于加速生成
        返回:
          logits: (batch, seq_len, vocab_size)
          presents: 新的 key/value 缓存列表
        """
        x = self.embed_tokens(input_ids)
        presents = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, past)
            presents.append(present)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, presents
