import torch
from model.gpt import GPT
from omegaconf import OmegaConf
from transformers import GPT2Tokenizer
from torch.nn import functional as F
import time, os
import random

# === 环境准备 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[+] Using {device} for training')

# === 加载配置和 tokenizer ===
cfg = OmegaConf.load("config/gpt_small.yaml")
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer/gpt_tokenizer")
vocab_size = tokenizer.vocab_size

# === 初始化模型和优化器 ===
model = GPT(**cfg.model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
start_step = 0
total_steps = cfg.train.total_steps

# === 获取训练 batch ===
with open("input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 统一拼接为一个长文本 → 然后 tokenized
tokenized = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]  # shape: [T]
num_tokens = tokenized.size(0)

def get_batch():
    ix = torch.randint(0, num_tokens - cfg.model.block_size - 1, (cfg.train.batch_size,))
    x = torch.stack([tokenized[i:i+cfg.model.block_size] for i in ix])
    y = torch.stack([tokenized[i+1:i+1+cfg.model.block_size] for i in ix])
    return x.to(device), y.to(device)


# === 训练 ===
compiled_model  = torch.compile(model)
compiled_model .train()

for step in range(start_step, total_steps):
    t0 = time.time()
    x, y = get_batch()
    logits = compiled_model (x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step:04d} | loss: {loss.item():.4f} | time: {time.time() - t0:.2f}s")

    if (step+1) % 1000 == 0:
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            'step': step+1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(ckpt_dir, f"gpt_step{step+1}.pt"))

# === 参数统计 ===
count = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {count / 1e6:.2f}M")
