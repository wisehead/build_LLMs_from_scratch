import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

print("Tokenization using gpt2")
print("--------------------------------")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))