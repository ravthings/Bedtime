import tiktoken
import mlx.core as mx
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token                        # 50256 = "end of story" token

ds = load_dataset("roneneldan/TinyStories", split="train")
stories = ds.select(range(100_000))["text"]

ids = []
for i, story in enumerate(stories):
    ids.extend(enc.encode(story))
    ids.append(EOT)                        # mark where each story ends
    if i % 10_000 == 0:
        print(f"{i} stories, {len(ids)} tokens")

mx.save("tokens_gpt2.npy", mx.array(ids))
print("saved:", len(ids), "tokens")