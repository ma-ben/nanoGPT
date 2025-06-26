import torch 
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.d_model = embed_dim // num_heads
        self.num_heads = num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, embed_dim = x.shape # (B, T, C)
        # 1. 计算q, k, v
        qkv = self.qkv_proj(x) # (B, T,3*C)
        # 2. 拆分q, k, v
        qkv = qkv.view(B, T, self.num_heads, 3*self.d_model)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, num_heads, d_model)*3
        # 3. 计算注意力分数
        q, k, v = [x.permute(0, 2, 1, 3) for x in(q, k, v)]  # (B, num_heads, T, d_model)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model) # (B, num_heads, T, T)
        # 4. 掩码
        mask = torch.tril(torch.ones(T, T, device=x.device)) # NOTE:mask必须动态创建，因为使用了变量T,NOTE:我意识到T其实不是动态的:)
        attn_score = attn_score.masked_fill(mask == 0, float('-inf')) # (B, num_heads, T, T)
        # 5. softmax + 对 v 加权
        attention_weights = torch.softmax(attn_score, dim=-1) # (B, num_heads, T, T)
        attn_outputs = torch.matmul(attention_weights, v) # (B, num_heads, T, d_model)
        # 6. 多头注意力concat回去
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).contiguous() # (B, T, num_heads, d_model)
        attn_outputs = attn_outputs.view(B, T, embed_dim)
        # 7. 最后一层映射
        return self.out_proj(attn_outputs)

class MLP(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4*embed_dim, bias=False)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim) # PreNorm before attention
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim) # PreNorm before MLP
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        # Attention段
        x = x + self.attn(self.ln1(x)) # 注意力前 LayerNorm，结果 residual
        # Feedforward段
        x = x + self.mlp(self.ln2(x)) # MLP 前 LayerNorm，结果 residual
        return x

# 模型定义：embedding → attention → linear output 
class GPT(nn.Module):
    def __init__(self, block_size, embed_dim, num_heads , num_layers, vocab_size = 2048):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(block_size, embed_dim),
            h = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(num_layers)]),
            ln_f = nn.LayerNorm(embed_dim),
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
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



