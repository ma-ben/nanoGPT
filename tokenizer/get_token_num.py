from transformers import AutoTokenizer
import os

tokenizer_path = "tokenizer/gpt_tokenizer"
# tokenizer_path = "tokenizer/llama_tokenizer"
file_path = "input.txt"

if not os.path.exists(tokenizer_path) and not os.path.exists(file_path):
    raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' or file '{file_path}' does not exist.")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text)
print(f"Total tokens: {len(ids)}")

