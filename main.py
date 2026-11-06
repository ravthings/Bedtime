import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import os
from mlx.utils import tree_flatten
import json
import tiktoken

import tiktoken
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab                   # 50257
EOT = enc.eot_token

def encode(s):
    return enc.encode(s)
def decode(ids):
    return enc.decode(ids)

data = mx.load("tokens_gpt2.npy")

# load the pre-encoded stories
data = mx.load("tokens.npy")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print("tokens:", len(data), "vocab_size:", vocab_size)

print("biggest token in data:", data.max().item())
print("data type:", data.dtype)

# hyperparameters
batch_size = 32
block_size = 128
n_embd = 128
save_file = "bedtime_llama.safetensors"

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
        self.rope  = nn.RoPE(head_size)                 # NEW
        self.head_size = head_size

    def __call__(self, x):
        B, T, C = x.shape
        k = self.rope(self.key(x))                   
        q = self.rope(self.query(x))                   
        v = self.value(x)                               
        

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
        hidden = 4 * n_embd * 2 // 3          # 341: keeps the size about the same as before
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))      # characters talk to each other
        x = x + self.ffwd(self.ln2(x))    # each character thinks on its own
        return x

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head=4) for _ in range(4)])# 4 blocks
        self.ln_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
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
print("parameters:", sum(v.size for _, v in tree_flatten(model.parameters())))
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
    if os.path.exists(save_file):
        model.load_weights(save_file)
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

    model.save_weights(save_file)
elif ask == "n":
    model.load_weights(save_file)
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
        if next_id.item() == EOT:
            break
        idx = mx.concatenate([idx, next_id], axis=1)
        print(decode([next_id.item()]), end="", flush=True)   # print each character as soon as it's made
    print()

while True:
    prompt = input("\nStart your story (or 'quit'): ")
    if prompt == "quit":
        break
    stream(model, prompt)