import mlx.core as mx

# load data
with open("stories.txt", "r", encoding="utf-8") as f:
    text = f.read()

# character-level tokenizer
vocab = sorted(set(text)) # every unique character
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[c] for c in s]
def decode(ids):
    return "".join(itos[i] for i in ids)

# data as an MLX array, split 90/10
data = mx.array(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# test
print("vocab_size:", vocab_size)
print(decode(encode("Once upon a time")))
print("data length:", len(data))