import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
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
            self.attn = SelfAttention(n_embed, n_head)
            self.out = nn.Linear(n_embed, vocab_size)

        def forward(self, idx):
            x = self.token_embed(idx)         # (B, T, C)
            x = self.attn(x)                  # (B, T, C)
            logits = self.out(x)              # (B, T, vocab_size)
            return logits

    model = TinyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    context = torch.tensor([[stoi["h"]]], device=device)
    model.eval()
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

    context = torch.tensor([[stoi["h"]]], device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            logits = model(context[:, -T:])  # 注意裁剪上下文
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_id], dim=1)
    print(decode(context[0].tolist()))
    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")






