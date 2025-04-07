from train import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} testing!!!')

# 打开tinyshakespeare
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

model = GPT2(vocab_size, block_size, n_embed, n_head, 4).to(device)
# 编译模型
model = torch.compile(model)  # Ensure the model is compiled for performance
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# 加载模型
resume = os.path.exists(model_name)
load_checkpoint(model, optimizer, model_name)
# 输出模型参数
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# 采样
print("Sample from the model: ")
model.eval()
context = torch.tensor([[stoi["h"]]], device=device)
with torch.no_grad():
    ans = model.generater(context,200)
print(decode(ans))

