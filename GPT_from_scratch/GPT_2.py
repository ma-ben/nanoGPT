import torch 
import torch.nn as nn
from TransformerBlock import TransformerBlock


class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.block = nn.Sequential(*[TransformerBlock(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # ✅ 权重共享
        self.lm_head.weight = self.token_embed.weight

    def forward(self, x): # x: (B, T)B, T = idx.size()
        B, T = x.size()
        pos = torch.arange(T, device=x.device)
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generater(self, context, max_len=100):
        for _ in range(max_len):
            logits = self(context[:, -self.pos_embed.num_embeddings:])  # 注意裁剪上下文
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)
        return context[0].tolist()


