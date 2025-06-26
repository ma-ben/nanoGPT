# train_gpt2_tokenizer.py ── 自制 GPT2-style Byte-Level BPE Tokenizer（50 257 词表）
# 依赖：pip install tokenizers transformers>=4.41

from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from transformers import GPT2TokenizerFast

# === 1. 训练 Byte-Level BPE（GPT2 默认 vocab_size=50257） ===
corpus = "input.txt"
out_dir = Path("tokenizer/gpt_tokenizer")
out_dir.mkdir(exist_ok=True)
bpe_tok = ByteLevelBPETokenizer()
bpe_tok.train(
    files=[corpus],
    vocab_size=8192,
    min_frequency=2,
    special_tokens=["", "<|pad|>", "<|unk|>"]
)
# 保存 vocab.json + merges.txt
bpe_tok.save_model(str(out_dir))

# === 2. 用 GPT2TokenizerFast 包装 HF 格式 ===
hf_tok = GPT2TokenizerFast(
    vocab_file=str(out_dir / "vocab.json"),
    merges_file=str(out_dir / "merges.txt"),
    bos_token="",
    eos_token="",
    pad_token="<|pad|>",
    unk_token="<|unk|>",
    add_prefix_space=True
)
hf_tok.save_pretrained(out_dir)

# === 3. 测试 ===
tok = GPT2TokenizerFast.from_pretrained("tokenizer/gpt_tokenizer")
sample = "今天天气不错，测试 GPT Tokenizer！"
print(tok.tokenize(sample))
print(tok(sample, return_tensors="pt").input_ids)
