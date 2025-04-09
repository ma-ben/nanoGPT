import torch
from model import GPT2
import os
from config import *
import threading

def async_save_checkpoint(model, optimizer, step, path=MODEL_FILE):
    def _save():
        torch.save({
            'model': model._orig_mod.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }, path)
    thread = threading.Thread(target=_save)
    thread.start()

def load_checkpoint(model, optimizer, path=MODEL_FILE):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    return step



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} testing!!!')

    # 测试样例


    # 打开tinyshakespeare
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    # 构造 batch
    def get_batch():
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # 初始化模型和优化器
    model = GPT2(vocab_size, block_size, n_embed, n_head, 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 训练和继续训练
    resume = os.path.exists(MODEL_FILE)
    start_step = 0
    if resume:
        start_step = load_checkpoint(model, optimizer, MODEL_FILE)
        print(f"✅ Resumed training from step {start_step}")
    else:
        print("🚀 Starting fresh training")
    
    # 训练循环
    from torch.nn import functional as F
    import time
    import json
    model = torch.compile(model)
    model.train()
    for step in range(start_step, total_steps):
        t0 = time.time()
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | loss: {loss.item():.4f}")
        if step % 500 == 0:
            async_save_checkpoint(model, optimizer, step)
            
    # 训练完成后，打印模型参数
    count = 0
    for param in model.parameters():
        count += param.numel()
    print(f"Total parameters: {count}")