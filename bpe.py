import json
import mlx.core as mx

with open("stories.txt", "r", encoding="utf-8") as f:
    text = f.read()[:200_000]# small slice to start, because pure Python is slow
ids = list(text.encode("utf-8"))# every character -> a byte (a number from 0 to 255)
print("starting length:", len(ids))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):# every neighbouring pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)         # found the pair -> swap in the new token
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


num_merges = 500# final vocab = 256 bytes + 500 new tokens = 756
merges = {}# (a, b) -> new token id: the "recipe" the tokenizer learns
vocab = {i: bytes([i]) for i in range(256)}# token id -> the bytes it stands for

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    new_id = 256 + i
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]          # join the bytes together

    if i % 50 == 0:
        word = vocab[new_id].decode("utf-8", errors="replace")
        print(f"merge {i}: {repr(word)}  (length now {len(ids)})")

print("final length:", len(ids))
print("compression:", round(200746 / len(ids), 2), "x")


def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # pick the pair that was learned EARLIEST (lowest new id)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                            # no more merges apply
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

s = "Once upon a time, Lily said hello to her dog."
tokens = encode(s)
print(tokens)
print("characters:", len(s), "→ tokens:", len(tokens))
print([decode([t]) for t in tokens])      # see how it split the sentence
print(decode(tokens) == s)      

with open("merges.json", "w") as f:
    json.dump([[a, b, new_id] for (a, b), new_id in merges.items()], f)
print("saved merges.json")

# 2. encode a big slice of stories, applying merges in the order they were learned
with open("stories.txt", "r", encoding="utf-8") as f:
    big_text = f.read()[:12_000_000]

big_ids = list(big_text.encode("utf-8"))
for i, (pair, new_id) in enumerate(merges.items()):
    big_ids = merge(big_ids, pair, new_id)
    if i % 50 == 0:
        print(f"encoding... merge {i}/{num_merges}, length {len(big_ids)}")

mx.save("tokens.npy", mx.array(big_ids))
print("saved tokens.npy:", len(big_ids), "tokens")

