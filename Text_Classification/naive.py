import math
import random


# Dataset

documents = [
    ("Win money now", "spam"),
    ("Limited offer win cash", "spam"),
    ("Meeting today", "not_spam"),
    ("Project meeting schedule", "not_spam")
]

docs_tokens = []
labels = []

for text, label in documents:
    tokens = text.lower().split()   
    docs_tokens.append(tokens)
    labels.append(1 if label == "spam" else 0)  


# Vocabulary

vocab = set()
for doc in docs_tokens:
    for word in doc:
        vocab.add(word)
vocab = list(vocab)



# TF
tf_docs = []
for doc in docs_tokens:
    tf = {}
    doc_len = len(doc)
    for word in vocab:
        tf[word] = doc.count(word) / doc_len
    tf_docs.append(tf)

# IDF
N = len(docs_tokens)
idf = {}
for word in vocab:
    count = sum(1 for doc in docs_tokens if word in doc)
    idf[word] = math.log(N / count)

# TF-IDF vector
tfidf_docs = []
for tf in tf_docs:
    vec = []
    for word in vocab:
        vec.append(tf[word] * idf[word])
    tfidf_docs.append(vec)


# Logistic Regression Function

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Initialize weights and bias

weights = [0.0] * len(vocab)
bias = 0.0
lr = 0.5
epochs = 1000


# Training with Gradient Descent

for epoch in range(epochs):
    for i in range(len(tfidf_docs)):
        x = tfidf_docs[i]
        y = labels[i]
        
      
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        pred = sigmoid(z)
        
       
        for j in range(len(weights)):
            weights[j] += lr * (y - pred) * x[j]
        bias += lr * (y - pred)


# Prediction Function

def predict_logreg(text):
    tokens = text.lower().split()
    x = [0.0] * len(vocab)
    for i, word in enumerate(vocab):
        if word in tokens:
            x[i] = 1.0  
    z = sum(w * xi for w, xi in zip(weights, x)) + bias
    prob = sigmoid(z)
    return 1 if prob >= 0.5 else 0


# Test Data 

test_texts = [
    "win cash now",          
    "meeting offer today",   
    "project win schedule",  
    "limited meeting"        
]

true_labels = [1, 0, 0, 1]  


# Predictions

predicted_labels = []
for text in test_texts:
    predicted_labels.append(predict_logreg(text))

print("Predicted Labels:", predicted_labels)


# Confusion Matrix

TP = TN = FP = FN = 0
for i in range(len(true_labels)):
    actual = true_labels[i]
    predicted = predicted_labels[i]
    
    if actual == 1 and predicted == 1:
        TP += 1
    elif actual == 0 and predicted == 0:
        TN += 1
    elif actual == 0 and predicted == 1:
        FP += 1
    elif actual == 1 and predicted == 0:
        FN += 1


# Accuracy

accuracy = (TP + TN) / (TP + TN + FP + FN)

print("\nConfusion Matrix")
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)
print("Accuracy:", accuracy)
