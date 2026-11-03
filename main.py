import mlx.core as mx
import mlx.nn as nn

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

# hyperparameters
batch_size = 32
block_size = 128 # 128 characters of context (about 2 sentences)
n_embd = 64

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = mx.random.randint(0, len(d) - block_size, (batch_size,)) # random start spots
    x = mx.stack([d[i : i + block_size] for i in ix.tolist()])
    y = mx.stack([d[i + 1 : i + block_size + 1] for i in ix.tolist()])
    return x, y

# test
xb, yb = get_batch("train")
print("x shape:", xb.shape)
print("first sequence:", decode(xb[0].tolist()))

#yhe model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)     # one row per character
        self.pos_emb = nn. Embedding(block_size, n_embd)     # one row per position
        self.lm_head = nn.Linear(n_embd, vocab_size)        # vector -> score for every character

    def __call__(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)                  # what each character is
        pos = self.pos_emb(mx.arange(T))         # where each character is
        x = tok + pos                            # combine both
        logits = self.lm_head(x)
        return logits

# test
model = TinyModel()
logits = model(xb)
print(logits.shape)
