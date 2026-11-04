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
