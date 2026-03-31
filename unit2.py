import spacy

nlp = spacy.load("en_core_web_sm")

text = "She enjoys playing tennis with her friends."
doc = nlp(text)

print("DEPENDENCY PARSING:\n")

for token in doc:
    print(f"{token.text} --> {token.dep_} --> {token.head.text}")

print("\nROOT WORD:")
for token in doc:
    if token.dep_ == "ROOT":
        print(token.text)