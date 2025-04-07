from config import *
import torch

# 构造 toy 数据
class Data:
    def __init__(self, data_path = "/home/taom/LLM-from-scratch/input.txt"): 
        if data_path == "dummy":
            self.text = "hello world!This is a test of the transformer block. Let's see how it works."
        else:

            with open(data_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
        self.B = batch_size
        self.T = block_size
        self.chars = sorted(set(self.text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)

    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, l): return ''.join([self.itos[i] for i in l])

    # 构造 batch
    def get_batch(self, device = "cuda" if torch.cuda.is_available() else "cpu"):
        ix = torch.randint(0, len(self.data) - self.T, (self.B,))
        x = torch.stack([self.data[i:i+self.T] for i in ix])
        y = torch.stack([self.data[i+1:i+self.T+1] for i in ix])
        return x.to(device), y.to(device)

if __name__ == "__main__":
    data = Data()
    x, y = data.get_batch()
    print(x)
    print(y)