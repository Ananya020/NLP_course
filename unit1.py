import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying a startup in India for 1 billion dollars."

# -------- Tokenization + POS --------
doc = nlp(text)

print("TOKENS & POS TAGS:")
for token in doc:
    print(token.text, "->", token.pos_)

# -------- Named Entity Recognition --------
print("\nNAMED ENTITIES:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

# -------- TF-IDF --------
documents = [text]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

print("\nTF-IDF KEYWORDS:")
scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

for word, score in sorted_scores:
    print(word, ":", round(score, 3))