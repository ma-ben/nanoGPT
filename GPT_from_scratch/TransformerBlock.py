import torch
import torch.nn as nn
from SelfAttention import SelfAttention

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

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed) # PreNorm before attention
        self.attn = SelfAttention(n_embed, n_heads)
        self.ln2 = nn.LayerNorm(n_embed) # PreNorm before MLP
        self.mlp = MLP(n_embed)

    def forward(self, x):
        # Attention段
        x = x + self.attn(self.ln1(x)) # 注意力前 LayerNorm，结果 residual
        # Feedforward段
        x = x + self.mlp(self.ln2(x)) # MLP 前 LayerNorm，结果 residual
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} testing!!!')

    # 测试样例
    # 定义超参数
    B = 16  # batch size
    T = 8   # context window
    n_embed = 32
    n_head = 4
    lr = 1e-2
    steps = 1000

    # 构造 toy 数据
    text = "hello world!This is a test of the transformer block. Let's see how it works."
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)

    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # 构造 batch
    def get_batch():
        ix = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i:i+T] for i in ix])
        y = torch.stack([data[i+1:i+T+1] for i in ix])
        return x.to(device), y.to(device)

    # 模型定义：embedding → attention → linear output
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, n_embed)
            self.attn = TransformerBlock(n_embed, n_head)
            self.out = nn.Linear(n_embed, vocab_size)

        def forward(self, idx):
            x = self.token_embed(idx)         # (B, T, C)
            x = self.attn(x)                  # (B, T, C)
            logits = self.out(x)              # (B, T, vocab_size)
            return logits

    model = TinyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.eval()
    context = torch.tensor([[stoi["h"]]], device=device)
    print("Sample from the model before training:")
    with torch.no_grad():
        for _ in range(100):
            logits = model(context[:, -T:])  # 注意裁剪上下文
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)
    print(decode(context[0].tolist()))
    model.train()
    # 训练循环
    from torch.nn import functional as F
    for step in range(steps):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step} | loss: {loss.item():.4f}")

    model.eval()
    context = torch.tensor([[stoi["h"]]], device=device)
    print("Sample from the model after training:",)
    with torch.no_grad():
        for _ in range(100):
            logits = model(context[:, -T:])  # 注意裁剪上下文
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)
    print(decode(context[0].tolist()))
    # 训练完成后，打印模型参数
    count = 0
    for param in model.parameters():
        count += param.numel()
    print(f"Total parameters: {count}")