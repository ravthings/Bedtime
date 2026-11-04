import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os

# load data
with open("stories.txt", "r", encoding="utf-8") as f:
    text = f.read()

# character-level tokenizer
vocab = sorted(set(text))
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

# hyperparameters
batch_size = 32
block_size = 128
n_embd = 64

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = mx.random.randint(0, len(d) - block_size, (batch_size,))
    x = mx.stack([d[i : i + block_size] for i in ix.tolist()])
    y = mx.stack([d[i + 1 : i + block_size + 1] for i in ix.tolist()])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)     # what each character has
        q = self.query(x)   # what each character is looking for
        v = self.value(x)   # what each character passes on

        scores = q @ k.transpose(0, 2, 1) * self.head_size ** -0.5
        mask = mx.tril(mx.ones((T, T))) == 1
        scores = mx.where(mask, scores, float("-inf"))
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]   # several heads side by side
        self.proj = nn.Linear(num_heads * head_size, n_embd)       # mix their outputs together

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)  # join the heads' outputs
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def __call__(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))      # characters talk to each other
        x = x + self.ffwd(self.ln2(x))    # each character thinks on its own
        return x

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head=4) for _ in range(4)])# 4 blocks
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(mx.arange(T))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]
        next_id = mx.random.categorical(logits, num_samples=1)
        idx = mx.concatenate([idx, next_id], axis=1)
    return idx

model = TinyModel()
mx.eval(model.parameters())
optimizer = optim.AdamW(learning_rate=1e-3)
loss_and_grad = nn.value_and_grad(model, loss_fn)

ask = input("Do you want to train the model? y/n ")

if ask == "y":
    epoche = input("How many epochs would you like to train for? ")
    epochs = int(epoche)
    learningr = input("What would you like the learning rate to be? ")
    lr = float(learningr)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.05)
    if os.path.exists("tiny.safetensors"):
        model.load_weights("tiny.safetensors")
        print("Resuming from saved model")
    else:
        print("No saved model, training from scratch")

    for epoch in range(epochs):
        xb, yb = get_batch("train")
        loss, grads = loss_and_grad(model, xb, yb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 250 == 0:
            xv, yv = get_batch("val")
            val_loss = loss_fn(model, xv, yv)
            print(f"epoch {epoch}: train {loss.item():.3f}, val {val_loss.item():.3f}")

    model.save_weights("tiny.safetensors")
elif ask == "n":
    model.load_weights("tiny.safetensors")
else:
    print("Invalid input")
    exit()

start = mx.array([encode("Once upon a time")])
print(decode(generate(model, start, 300)[0].tolist()))

def stream(model, prompt, max_new_tokens=500):
    idx = mx.array([encode(prompt)])
    print(prompt, end="", flush=True)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])[:, -1, :]
        next_id = mx.random.categorical(logits, num_samples=1)
        idx = mx.concatenate([idx, next_id], axis=1)
        print(decode([next_id.item()]), end="", flush=True)   # print each character as soon as it's made
    print()

while True:
    prompt = input("\nStart your story (or 'quit'): ")
    if prompt == "quit":
        break
    stream(model, prompt)