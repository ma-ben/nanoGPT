from transformers import GPT2Tokenizer
from datasets import Dataset

# 1. 读取并编码文本
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer/gpt_tokenizer")
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text)
print(f"Total tokens: {len(ids)}")
# block_size = 128
# input_ids = []
# labels = []

# # 2. 滑窗切块
# for i in range(0, len(ids) - block_size, block_size):
#     chunk = ids[i: i + block_size + 1]
#     input_ids.append(chunk[:-1])
#     labels.append(chunk[1:])

# # 3. 转换成 HuggingFace Dataset
# dataset = Dataset.from_dict({
#     "input_ids": input_ids,
#     "labels": labels
# })
