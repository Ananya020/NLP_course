import streamlit as st
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import gensim.downloader as api

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')   

# Load models
nlp = spacy.load("en_core_web_sm")
w2v_model = api.load("glove-wiki-gigaword-50")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Streamlit UI
st.title("🧠 Smart Text Analyzer")
st.write("Analyze text using NLP concepts")

text = st.text_area("Enter your text:")

if text:

    # -----------------------------
    # Tokenization
    # -----------------------------
    tokens = word_tokenize(text)
    st.subheader("🔹 Tokens")
    st.write(tokens)

    # -----------------------------
    # Stemming & Lemmatization
    # -----------------------------
    stems = [stemmer.stem(word) for word in tokens]
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]

    st.subheader("🔹 Stemming vs Lemmatization")
    st.write("Stems:", stems)
    st.write("Lemmas:", lemmas)

    # -----------------------------
    # POS Tagging & NER (spaCy)
    # -----------------------------
    doc = nlp(text)

    st.subheader("🔹 POS Tagging")
    pos_tags = [(token.text, token.pos_) for token in doc]
    st.write(pos_tags)

    st.subheader("🔹 Named Entity Recognition")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(entities)

    # -----------------------------
    # TF-IDF
    # -----------------------------
    st.subheader("🔹 TF-IDF Keywords")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    st.write(sorted_scores[:10])

    # -----------------------------
    # N-grams
    # -----------------------------
    st.subheader("🔹 Bigrams")

    bigrams = list(nltk.ngrams(tokens, 2))
    st.write(bigrams)

    # -----------------------------
    # Word Frequency
    # -----------------------------
    st.subheader("🔹 Word Frequency")

    freq = Counter(tokens)
    st.write(freq)

    # -----------------------------
    # Word Similarity (Word2Vec)
    # -----------------------------
    st.subheader("🔹 Word Similarity")

    word = st.text_input("Enter a word to find similar words:")

    if word:
        if word in w2v_model:
            similar_words = w2v_model.most_similar(word, topn=5)
            st.write(similar_words)
        else:
            st.write("Word not found in vocabulary")