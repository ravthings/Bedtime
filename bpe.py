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

stats = get_stats(ids)
top = max(stats, key=stats.get)
print("most common pair:", repr(bytes(top).decode("utf-8")), "seen", stats[top], "times")

ids = merge(ids, top, 256)# 256 = first new token (0–255 are taken by bytes)
print("after one merge:", len(ids))

num_merges = 500# final vocab = 256 bytes + 500 new tokens = 756
merges = {}# (a, b) -> new token id: the "recipe" the tokenizer learns
vocab = {i: bytes([i]) for i in range(256)}# token id -> the bytes it stands for


