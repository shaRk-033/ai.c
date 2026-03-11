import tiktoken
import urllib.request

# download tiny shakespeare
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
print("downloading shakespeare...")
data = urllib.request.urlopen(url).read().decode("utf-8")
print(f"downloaded {len(data)} chars")

# tokenize with gpt2 tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
print(f"tokenized into {len(tokens)} tokens")

# write comma separated tokens to output.txt
with open("output.txt", "w") as f:
    f.write(",".join(str(t) for t in tokens))

print("wrote tokens to output.txt")
