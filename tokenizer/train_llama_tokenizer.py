import sentencepiece as spm
from pathlib import Path
from transformers import LlamaTokenizerFast

corpus = "input.txt"                 # 你的语料
out_dir = Path("tokenizer/llama_tokenizer")
out_dir.mkdir(exist_ok=True)

# ① SentencePiece 训练；byte_fallback=True 会自动注入 <0x00>…<0xFF>
spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=str(out_dir / "spm"),  # 会生成 spm.model / spm.vocab
    vocab_size=8192,                   # LLaMA 默认 32 k
    model_type="bpe",                   # LLaMA 是 BPE+Unigram 混；实测 bpe OK
    character_coverage=1.0,             # 全字符覆盖
    byte_fallback=True,                 # 关键！生成 256 个字节回退 token
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=[],            # 你若想加 <mask> 等在这放
)

# ② HF 封装；LlamaTokenizerFast 能直接吃 spm.model
tok = LlamaTokenizerFast(
    vocab_file=str(out_dir / "spm.model"),  # transformers 会自动找 .vocab
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
)
tok.save_pretrained(out_dir)             # 写 tokenizer_config.json / special_tokens_map.json

# ③ 测试
sample = "今天天气不错，测试 LLaMA Tokenizer！"
print(tok.tokenize(sample))
print(tok(sample, return_tensors="pt").input_ids)
