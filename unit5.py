import re
import math
from collections import Counter, defaultdict

# =============================================================================
# 1. RETRIEVAL-BASED CHATBOT (TF-IDF & COSINE SIMILARITY)
# =============================================================================

def tokenize(text): 
    return re.findall(r'\b\w+\b', text.lower())

def tfidf_similarity(q_tok, d_tok):
    terms = set(q_tok) | set(d_tok)
    q, d  = Counter(q_tok), Counter(d_tok)
    dot   = sum(q[t] * d[t] for t in terms)
    qn    = math.sqrt(sum(v**2 for v in q.values()))
    dn    = math.sqrt(sum(v**2 for v in d.values()))
    return dot / (qn * dn) if qn and dn else 0.0

def run_chatbot_demo():
    print("\n--- Task 1: Retrieval-Based Chatbot ---")
    corpus = [
        ("greeting", ["hello", "hi", "hey"], "Hello! How can I help you today?"),
        ("nlp", ["what is nlp", "natural language processing"], "NLP enables computers to understand human language."),
        ("transformer", ["transformer", "attention", "bert"], "Transformers use self-attention for long-range dependencies."),
        ("default", [], "I can help with NLP, transformers, and RNNs.")
    ]
    
    queries = ["Hello there!", "Explain natural language processing"]
    for q in queries:
        q_tok = tokenize(q)
        best_score, best_resp = 0.0, corpus[-1][2]
        for _, patterns, resp in corpus[:-1]:
            for pat in patterns:
                score = tfidf_similarity(q_tok, tokenize(pat))
                if score > best_score:
                    best_score, best_resp = score, resp
        print(f"User: {q}\nBot : {best_resp} [Score: {best_score:.4f}]\n")

# =============================================================================
# 2. EXTRACTIVE QUESTION ANSWERING
# =============================================================================

def sent_split(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) > 3]

def run_qa_demo():
    print("--- Task 2: Extractive Question Answering ---")
    context = ("Natural Language Processing (NLP) is a subfield of AI. "
               "Common NLP tasks include text classification, NER, and QA. "
               "Deep learning has significantly improved NLP performance.")
    
    question = "What are the common tasks in NLP?"
    sentences = sent_split(context)
    # Re-using the similarity logic from Task 1
    scored = [(tfidf_similarity(tokenize(question), tokenize(s)), s) for s in sentences]
    score, ans = max(scored, key=lambda x: x[0])
    
    print(f"Context: {context[:60]}...")
    print(f"Q: {question}\nA: {ans} [Confidence: {score:.4f}]\n")

# =============================================================================
# 3. TEXT SUMMARIZATION (EXTRACTIVE & ABSTRACTIVE)
# =============================================================================

STOP = {"the","a","an","is","are","was","of","in","on","at","to","for","and","it"}

def extractive_summary(text, ratio=0.5):
    sents = sent_split(text)
    words = [w for w in tokenize(text) if w not in STOP]
    freq = Counter(words)
    max_f = max(freq.values()) if freq else 1
    freq = {w: f/max_f for w, f in freq.items()}
    
    scores = {i: sum(freq.get(w, 0) for w in tokenize(s)) for i, s in enumerate(sents)}
    n = max(1, round(len(sents) * ratio))
    top_indices = sorted(sorted(scores, key=scores.get, reverse=True)[:n])
    return ' '.join(sents[i] for i in top_indices)

def abstractive_summary(text):
    # Template-based approach
    nouns = re.findall(r'\b[A-Z][a-z]+\b', text)[:1]
    subj = nouns[0] if nouns else "The text"
    keywords = [w for w, _ in Counter(tokenize(text)).most_common(3) if w not in STOP]
    return f"{subj} explores concepts involving {', '.join(keywords)}."

def run_summarization_demo():
    print("--- Task 3: Text Summarization ---")
    doc = ("Climate change refers to long-term shifts in global temperatures. "
           "Human activities have driven emissions since the industrial revolution. "
           "Burning fossil fuels releases CO2, causing global warming.")
    
    print(f"Original Text: {doc}")
    print(f"Extractive  : {extractive_summary(doc)}")
    print(f"Abstractive : {abstractive_summary(doc)}\n")

# =============================================================================
# 4. MACHINE TRANSLATION & BLEU EVALUATION
# =============================================================================

EN_FR = {"machine": "machine", "translation": "traduction", "is": "est", "a": "un", "natural": "naturel"}

def run_translation_demo():
    print("--- Task 4: Machine Translation & BLEU ---")
    # Dictionary Translation
    source = "Machine translation is a natural task"
    translated = ' '.join(EN_FR.get(w.lower(), w) for w in source.split())
    
    # BLEU-1 Calculation
    def bleu1(hyp, ref):
        h_tok, r_tok = hyp.lower().split(), set(ref.lower().split())
        precision = sum(1 for w in h_tok if w in r_tok) / len(h_tok)
        return precision

    hyp = "Machine traduction est un natural task"
    ref = "Machine translation is a natural task"
    
    print(f"Source   : {source}")
    print(f"FR Trans : {translated}")
    print(f"BLEU-1 Score: {bleu1(hyp, ref):.4f}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("      NLP PRACTICE PROGRAMS - UNIT 5 COMPLETE")
    print("="*60)
    run_chatbot_demo()
    run_qa_demo()
    run_summarization_demo()
    run_translation_demo()
