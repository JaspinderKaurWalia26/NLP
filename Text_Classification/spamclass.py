import math
"""
Task: Build Spam Classifier
Preprocess text
Convert to TF-IDF
Train Naive Bayes
Predict spam/not spam
"""
# Spam Classification using Naive bayes

# Sample Dataset
documents = [
    ("Win money now", "spam"),
    ("Limited offer win cash", "spam"),
    ("Meeting today", "not_spam"),
    ("Project meeting schedule", "not_spam")
]

# separating text and labels (input,output)
docs_tokens = []
labels = []

for text, label in documents:
    tokens = text.lower().split()   
    docs_tokens.append(tokens)
    labels.append(label)


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


# TF-IDF

tfidf_docs = []
for tf in tf_docs:
    tfidf = {}
    for word in vocab:
        tfidf[word] = tf[word] * idf[word]
    tfidf_docs.append(tfidf)


# Split by Class

spam_tfidf = []
not_spam_tfidf = []

for i in range(len(labels)):
    if labels[i] == "spam":
        spam_tfidf.append(tfidf_docs[i])
    else:
        not_spam_tfidf.append(tfidf_docs[i])


# Calculating Prior Probabilities
# P_spam= count of spam labels / total length of data
P_spam = len(spam_tfidf) / len(tfidf_docs)
P_not_spam = len(not_spam_tfidf) / len(tfidf_docs)



# Average TF-IDF per Class

def avg_tfidf(docs):
    avg = {}
    for word in vocab:
        total = 0
        for doc in docs:
            total += doc[word]
        avg[word] = (total + 1e-6) / len(docs)  
    return avg

spam_avg = avg_tfidf(spam_tfidf)
not_spam_avg = avg_tfidf(not_spam_tfidf)


# Prediction Function

def predict_tfidf(text):
    words = text.lower().split()
    spam_score = math.log(P_spam)
    not_spam_score = math.log(P_not_spam)
    
    for word in words:
        if word in vocab:
            spam_score += math.log(spam_avg[word])
            not_spam_score += math.log(not_spam_avg[word])
    
    return "SPAM" if spam_score > not_spam_score else "NOT SPAM"


# Test Data 

test_texts = [
    "win cash now",          
    "meeting offer today",   
    "project win schedule",  
    "limited meeting"        
]

true_labels = [
    "spam",
    "not_spam",
    "not_spam",
    "spam"
]



# Predictions

predicted_labels = []
for text in test_texts:
    predicted_labels.append(predict_tfidf(text))

print("Predicted Labels:", predicted_labels)


# Confusion Matrix

TP = TN = FP = FN = 0

for i in range(len(true_labels)):
    actual = true_labels[i]
    predicted = predicted_labels[i]
    
    if actual == "spam" and predicted == "SPAM":
        TP += 1
    elif actual == "not_spam" and predicted == "NOT SPAM":
        TN += 1
    elif actual == "not_spam" and predicted == "SPAM":
        FP += 1
    elif actual == "spam" and predicted == "NOT SPAM":
        FN += 1


# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("\nConfusion Matrix")
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)
print("Accuracy:", accuracy)
