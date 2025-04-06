import torch
from torch import nn





class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.split_size = config.n_embd * 3
        self.c_attn = nn.Linear(config.n_embd, self.split_size)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.attention_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.c_attn.NANOGPT_SCALE_INIT = 1
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).view(B, T, self.n_head, 3 * C // self.n_head)
        qkv = qkv.permute(0, 2, 1, 3).contiguous()
        q, k, v = qkv.split(C // self.n_head, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (C ** -0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)
        y = torch.matmul(attn, v)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.attn(self.ln_2(x))
        return x













