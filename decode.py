import tiktoken, sys

enc = tiktoken.get_encoding("gpt2")
# pass comma separated token ids as argument, or pipe them in
raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
tokens = [int(t.strip()) for t in raw.split(",") if t.strip()]
print(enc.decode(tokens))
