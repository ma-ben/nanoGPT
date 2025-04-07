import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()

        assert n_embed % n_head == 0
        self.d_model = n_embed // n_head
        self.n_head = n_head
        
        self.qkv_proj = nn.Linear(n_embed, 3*n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        B, T, n_embed = x.shape # (B, T, C)
        # 1. 计算q, k, v
        qkv = self.qkv_proj(x) # (B, T,3*C)
        # 2. 拆分q, k, v
        qkv = qkv.view(B, T, self.n_head, 3*self.d_model)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, n_head, d_model)*3
        # 3. 计算注意力分数
        q, k, v = [x.permute(0, 2, 1, 3) for x in(q, k, v)]  # (B, n_head, T, d_model)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model) # (B, n_head, T, T)
        # 4. 掩码
        mask = torch.tril(torch.ones(T, T, device=x.device)) # NOTE:mask必须动态创建，因为使用了变量T,NOTE:我意识到T其实不是动态的:)
        attn_score = attn_score.masked_fill(mask == 0, float('-inf')) # (B, n_head, T, T)
        # 5. softmax + 对 v 加权
        attention_weights = torch.softmax(attn_score, dim=-1) # (B, n_head, T, T)
        attn_outputs = torch.matmul(attention_weights, v) # (B, n_head, T, d_model)
        # 6. 多头注意力concat回去
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous() # (B, T, n_head, d_model)
        attn_outputs = attn_outputs.view(B, T, n_embed)
        # 7. 最后一层映射
        return self.out_proj(attn_outputs)

