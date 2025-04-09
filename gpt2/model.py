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

class MLP(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4*n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*n_embed, n_embed)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed) # PreNorm before attention
        self.attn = MultiHeadAttention(n_embed, n_heads)
        self.ln2 = nn.LayerNorm(n_embed) # PreNorm before MLP
        self.mlp = MLP(n_embed)

    def forward(self, x):
        # Attention段
        x = x + self.attn(self.ln1(x)) # 注意力前 LayerNorm，结果 residual
        # Feedforward段
        x = x + self.mlp(self.ln2(x)) # MLP 前 LayerNorm，结果 residual
        return x

# 模型定义：embedding → attention → linear output 
class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed),
            wpe = nn.Embedding(block_size, n_embed),
            h = nn.ModuleList([Block(n_embed, n_head) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embed),
        ))
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, x): # x: (B, T)B, T = idx.size()
        B, T = x.size()
        pos = torch.arange(T, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(x)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate_stream(self, context, max_len=100):
        for _ in range(max_len):
            logits = self(context[:, -self.transformer.wpe.num_embeddings:])  # 裁剪上下文窗口
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)
            yield next_id.item()  # 流式输出



