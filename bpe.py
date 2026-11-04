with open("stories.txt", "r", encoding="utf-8") as f:
    text = f.read()[:200_000]# small slice to start, because pure Python is slow

