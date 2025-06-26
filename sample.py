import torch
import os
from omegaconf import OmegaConf
from model.gpt import GPT
from transformers import GPT2Tokenizer

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference!!!')
checkpoint_path = "checkpoints/gpt_step500.pt"  # 你需要替换为实际的检查点路径

# 加载 config 和 tokenizer
cfg = OmegaConf.load("config/gpt_small.yaml")
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer/gpt_tokenizer") # 应该是你训练时保存的tokenizer路径
model = GPT(**cfg.model).to(device)

# 加载模型参数（可选）
if os.path.exists(checkpoint_path):
    print(f"Loading model from {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model"])

# 输出模型规模
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# 模型进入 eval 模式
model.eval()

# 输入 prompt，自动编码
prompt = "Hello"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 生成（流式输出）
print("\nGenerated:")
with torch.no_grad():
    for token_id in model.generate_stream(input_ids, max_len=200):
        print(tokenizer.decode([token_id]), end="", flush=True)
print()