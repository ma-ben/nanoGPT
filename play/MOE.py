import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb

# 初始化 wandb
wandb.init(project="moe-routing-demo", name="routing-visual", mode="online")

class Config:
    hidden_size = 8
    num_experts_per_tok = 2
    n_routed_experts = 4
    norm_topk_prob = True

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx.view(bsz, seq_len, -1), topk_weight.view(bsz, seq_len, -1)

# 模拟数据
config = Config()
gate = MoEGate(config)
dummy_hidden = torch.randn(1, 8, config.hidden_size)  # batch=1, seq_len=8
topk_idx, topk_weight = gate(dummy_hidden)

# wandb 可视化 token → expert 路由
for token_pos in range(topk_idx.shape[1]):
    for k in range(config.num_experts_per_tok):
        wandb.log({
            "token_position": token_pos,
            "expert_id": int(topk_idx[0, token_pos, k]),
            "routing_weight": float(topk_weight[0, token_pos, k]),
        })

print("🎉 路由信息已发送到 wandb！打开网页查看图表 👇")
print("https://wandb.ai/你的用户名/moe-routing-demo")

wandb.finish()
