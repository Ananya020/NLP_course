import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (BertTokenizer, BertForSequenceClassification, 
                          RobertaTokenizer, RobertaForSequenceClassification,
                          Trainer, TrainingArguments)

# =============================================================================
# 1. VANILLA RNN CHARACTER-LEVEL LANGUAGE MODEL (FROM SCRATCH)
# =============================================================================

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.hidden_size = hidden_size
        self.lr = lr
        # Xavier-like initialization
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.hs, self.xs = {}, {}
        self.hs[-1] = h.copy()
        ys = []
        for t, x in enumerate(inputs):
            self.xs[t] = x.reshape(-1, 1)
            h = np.tanh(self.Wxh @ self.xs[t] + self.Whh @ h + self.bh)
            self.hs[t] = h
            y = self.Why @ h + self.by
            ys.append(y)
        return ys, h

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def cross_entropy(self, probs, target):
        return -np.log(probs[target, 0] + 1e-8)

def run_task_1():
    print("\n--- Task 1: Vanilla RNN Character Model ---")
    text = "hello world nlp rnn"
    chars = sorted(set(text))
    ch2ix = {c: i for i, c in enumerate(chars)}
    ix2ch = {i: c for c, i in ch2ix.items()}
    V = len(chars)

    rnn = SimpleRNN(input_size=V, hidden_size=16, output_size=V, lr=0.1)
    
    def one_hot(idx, size):
        v = np.zeros((size, 1))
        v[idx] = 1
        return v

    for epoch in range(5):
        total_loss = 0
        for i in range(len(text)-1):
            x = one_hot(ch2ix[text[i]], V)
            t = ch2ix[text[i+1]]
            ys, _ = rnn.forward([x])
            probs = rnn.softmax(ys[0])
            total_loss += rnn.cross_entropy(probs, t)
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/(len(text)-1):.4f}")

    # Generation
    seed = 'h'
    h = np.zeros((rnn.hidden_size, 1))
    result = seed
    for _ in range(20):
        x = one_hot(ch2ix[seed], V)
        ys, h = rnn.forward([x])
        probs = rnn.softmax(ys[0]).ravel()
        idx = np.random.choice(V, p=probs)
        seed = ix2ch[idx]
        result += seed
    print(f"Generated text: {result}")

# =============================================================================
# 2. LSTM CELL & SENTIMENT ANALYSIS (FROM SCRATCH)
# =============================================================================

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        d = hidden_dim
        self.W = np.random.randn(4*d, input_dim + d) * 0.01
        self.b = np.zeros((4*d, 1))

    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
    def tanh(self, x):    return np.tanh(np.clip(x, -20, 20))

    def forward(self, x, h_prev, c_prev):
        d = h_prev.shape[0]
        combo = np.vstack([x, h_prev])
        gates = self.W @ combo + self.b
        i = self.sigmoid(gates[:d])
        f = self.sigmoid(gates[d:2*d])
        g = self.tanh(gates[2*d:3*d])
        o = self.sigmoid(gates[3*d:])
        c = f * c_prev + i * g
        h = o * self.tanh(c)
        return h, c, {"i": i, "f": f, "g": g, "o": o}

class LSTMSentiment:
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16):
        self.embed = np.random.randn(vocab_size, embed_dim) * 0.1
        self.cell = LSTMCell(embed_dim, hidden_dim)
        self.Wout = np.random.randn(1, hidden_dim) * 0.01
        self.bout = np.zeros((1, 1))
        self.hidden_dim = hidden_dim

    def forward(self, token_ids):
        h = np.zeros((self.hidden_dim, 1))
        c = np.zeros((self.hidden_dim, 1))
        all_gates = []
        for idx in token_ids:
            x = self.embed[idx].reshape(-1, 1)
            h, c, gates = self.cell.forward(x, h, c)
            all_gates.append(gates)
        logit = self.Wout @ h + self.bout
        prob = 1 / (1 + np.exp(-logit))
        return prob[0, 0], h, all_gates

def run_task_2():
    print("\n--- Task 2: LSTM Sentiment Analysis ---")
    vocab = {"good":0,"great":1,"excellent":2,"bad":3,"terrible":4,"movie":5,"film":6,
             "the":7,"was":8,"is":9,"very":10,"sad":11,"happy":12,"awful":13,"amazing":14,"boring":15}
    sentences = [("the movie was great and amazing", 1), ("the film was terrible and awful", 0)]
    
    model = LSTMSentiment(vocab_size=len(vocab))
    for text, label in sentences:
        ids = [vocab.get(w, 0) for w in text.split()]
        prob, _, gates = model.forward(ids)
        pred = "POSITIVE" if prob > 0.5 else "NEGATIVE"
        print(f"Text: {text} | Pred: {pred} (Conf: {prob:.3f} if POS else {1-prob:.3f})")

# =============================================================================
# 3. BERT FINE-TUNING (TRANSFORMERS)
# =============================================================================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def run_task_3():
    print("\n--- Task 3: BERT Sentiment Inference ---")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    test_texts = ["An amazing cinematic experience!", "Worst movie ever."]
    for text in test_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = model(**enc)
            pred = torch.argmax(out.logits, dim=1).item()
        print(f"Text: {text} | Prediction: {'POS' if pred else 'NEG'}")

# =============================================================================
# 4. ROBERTA TOPIC CLASSIFICATION & TOKENIZATION COMPARISON
# =============================================================================

def run_task_4():
    print("\n--- Task 4: RoBERTa vs BERT Tokenization ---")
    sample = "RoBERTa uses Byte-Pair Encoding"
    
    b_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    r_tok = RobertaTokenizer.from_pretrained("roberta-base")
    
    print(f"BERT Tokens:    {b_tok.tokenize(sample)}")
    print(f"RoBERTa Tokens: {r_tok.tokenize(sample)}")
    
    # Simple Inference Demo
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
    labels = ["Technology", "Sports", "Health"]
    text = "The new GPU achieves record-breaking performance."
    enc = r_tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
        pred = torch.argmax(out.logits, dim=1).item()
    print(f"Input: {text}\nPredicted Topic: {labels[pred]}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
