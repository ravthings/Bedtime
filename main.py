import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)# one attention head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(mx.arange(T))
        x = self.sa_head(x)#characters look at each other
        logits = self.lm_head(x)
        return logits

# test
model = TinyModel()
logits = model(xb)
print(logits.shape)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)# what each character has
        q = self.query(x)# what each character is looking for
        v = self.value(x) # what each character passes on

        scores = q @ k.transpose(0, 2, 1) * self.head_size ** -0.5   # (B, T, T)

        mask = mx.tril(mx.ones((T, T))) == 1
        scores = mx.where(mask, scores, float("-inf"))

        weights = mx.softmax(scores, axis=-1)

        out = weights @ v
        return out

def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")

model = TinyModel()
mx.eval(model.parameters())
optimizer = optim.AdamW(learning_rate=1e-3)
loss_and_grad = nn.value_and_grad(model, loss_fn)

ask = input("Do you you want to train the model? y/n ")

if ask == "y":
    for step in range(3000):
        xb, yb = get_batch("train")                    # new batch every step
        loss, grads = loss_and_grad(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)   # "do the maths now"

        if step % 250 == 0:
            xv, yv = get_batch("val")
            val_loss = loss_fn(model, xv, yv)
            print(f"step {step}: train {loss.item():.3f}, val {val_loss.item():.3f}")

    model.save_weights("tiny.safetensors")
