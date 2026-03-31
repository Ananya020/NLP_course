import gensim.downloader as api

# Load pretrained model
model = api.load("glove-wiki-gigaword-50")

word = "king"

print(f"Words similar to '{word}':\n")

similar_words = model.most_similar(word, topn=5)

for w, score in similar_words:
    print(w, ":", round(score, 3))