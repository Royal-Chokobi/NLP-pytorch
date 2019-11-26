import spacy

def n_grams(x, n):
    for i in range(len(x)-n+1):
        print([x[i:i+n]])

def token_sb(doc):
    for token in doc:
        print('{} -> {} / {}'.format(token, token.lemma_, token.pos_))


npl = spacy.load('en_core_web_sm')
txt = "hello world, In this python and pytorch :)"
text = npl(txt)

print(list(text))
print(text)

n_grams(text,3)

doc = npl("he was running late fly, flew, flew, flown")

token_sb(doc)

doc1 = npl("Marry slapped the green witch")
for chunk in doc1.noun_chunks:
    print("chunk label : {} -> {}".format(chunk, chunk.label_))
