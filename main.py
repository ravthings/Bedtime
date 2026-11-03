import mlx
from datasets import load_dataset

from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train")
print("total stories:", len(ds))

n = 20000 # start small
stories = ds.select(range(n))["text"]

with open("stories.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(stories))

print(f"saved {n} stories")
print(stories[0])