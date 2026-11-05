# Bedtime 📖

A small transformer that writes short children's stories, built from scratch in **MLX** on Apple Silicon.

Give it the start of a story like, Once upon a time, and it continues it, character by character.

## Demo


https://github.com/user-attachments/assets/65db3cf9-95e5-4d05-9f02-d23208caadb9



## Example

> Once upon a time, there was a basket. The terrible bird had to pup in the box and went found at the lion... The sun smiled and waved to out hara laugh.

## How it works

- **Data:** 20,000 stories from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
- **Tokenizer:** BPE tokenizer
- **Model:** a decoder-only transformer
  - token and position embeddings
  - 4 blocks, each with 4-head self-attention and a feed-forward layer
  - residual connections and LayerNorm
  - embedding size [64 / 128], context of 128 characters
  - about 1 000 000 parameters
- **Training:** AdamW (learning rate 0.001, weight decay 0.05), 5,000 steps on a Mac

## Results

| 3 | 0.96 |
|---|---|
| One attention head | 2.16 |
| 4 blocks × 4 heads | **0.95** |

Going from one attention head to a full transformer more than halved the loss. The model learned real sentences, character names and story structure.

## How to run

```
pip install mlx datasets
python get_data.py
python train.py
```

Choose `y` to train (it carries on from the saved model if there is one) or `n` to load the saved model. Then type the start of a story and watch it continue. Type `quit` to stop.

## Limitations

- It only sees 128 characters at a time, so stories wander off topic.
- Because it works letter by letter, it sometimes invents words.

## Next steps

- A BPE tokenizer, so it works with word pieces instead of letters
- A bigger model and longer context

## Credits
TinyStories by Eldan & Li.
