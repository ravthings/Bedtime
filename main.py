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
        self.proj = nn.Linear(num_heads * head_size, n_embd, bias=False)     # mix their outputs together

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

    def __call__(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.tok_emb.as_linear(x)

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

ask = input("Train (y), load (n), or export (e)? ")

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

elif ask == "e":
    import json
    model.load_weights(save_file)
    w = dict(tree_flatten(model.parameters()))
    hf = {
        "model.embed_tokens.weight": w["tok_emb.weight"],
        "model.norm.weight": w["ln_f.weight"],
    }
    for i in range(4):
        p, h = f"blocks.layers.{i}.", f"model.layers.{i}."
        hf[h + "input_layernorm.weight"] = w[p + "ln1.weight"]
        hf[h + "post_attention_layernorm.weight"] = w[p + "ln2.weight"]
        for ours, theirs in [("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")]:
            # stack the 4 small head layers into one big layer, in order
            hf[h + f"self_attn.{theirs}.weight"] = mx.concatenate(
                [w[p + f"sa.heads.{j}.{ours}.weight"] for j in range(4)], axis=0)
        hf[h + "self_attn.o_proj.weight"] = w[p + "sa.proj.weight"]
        for name in ["gate_proj", "up_proj", "down_proj"]:
            hf[h + f"mlp.{name}.weight"] = w[p + f"ffwd.{name}.weight"]

    os.makedirs("export", exist_ok=True)
    mx.save_safetensors("export/model.safetensors", hf, metadata={"format": "pt"})

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": n_embd,
        "intermediate_size": 4 * n_embd * 2 // 3,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": n_embd // 4,
        "vocab_size": vocab_size,
        "max_position_embeddings": block_size,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "bos_token_id": EOT,
        "eos_token_id": EOT,
        "torch_dtype": "float32",
    }
    with open("export/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("exported", len(hf), "tensors to export/")
    exit()
    
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