# Converting text into Bow
# Sample documents
docs = ['I love NLP', 'NLP is amazing', 'I love coding']

# Building vocabulary
words = set()
for doc in docs:
    tokens = doc.split()
    words.update(tokens)
# converting set to list to keep order
words = list(words) 
print("Vocabulary:", words)

# Create Bag of Words vectors
bow_vectors = []

for doc in docs:
    tokens = doc.split()
    vector = []
    for word in words:
        vector.append(tokens.count(word))  # count occurrences
    bow_vectors.append(vector)

# Step 3: Print vectors
for i, vec in enumerate(bow_vectors):
    print(f"Document {i+1} BoW:", vec)
    
# Converting text into TF-IDF
 
documents = [
    "I love NLP and Machine Learning",
    "Machine Learning is amazing",
    "I love learning new things"
]
docs_tokens = [doc.lower().split() for doc in documents]
print(docs_tokens)

vocab = sorted(set(word for doc in docs_tokens for word in doc))
print(vocab)


tf_docs = []

for doc in docs_tokens:              
    tf_doc = {}                      
    doc_len = len(doc)        
   
    for word in vocab:  
        tf_doc[word] = doc.count(word) / doc_len 
    tf_docs.append(tf_doc) 
print(tf_docs)
import math

N = len(docs_tokens)  
idf = {}

for word in vocab:
    count = sum(1 for doc in docs_tokens if word in doc)
    if count > 0:  
        idf[word] = math.log(N / count)
    else:
        idf[word] = 0  
        

print(idf)
tfidf_docs = []

for tf_doc in tf_docs:
    tfidf_doc = {}
    for word in vocab:
        tfidf_doc[word] = tf_doc[word] * idf[word]
    tfidf_docs.append(tfidf_doc)

for i, doc in enumerate(tfidf_docs):
    print(f"Document {i+1}: {doc}")

