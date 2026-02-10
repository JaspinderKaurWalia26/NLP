"""
Explain why cosine similarity is better than Euclidean for text?
Cosine similarity is better for text because it ignores length and focuses on meaning. Even if one text is short and another is long, cosine looks at the direction of their vectors (the ideas) rather than their size. It gives a simple score showing how similar the texts are, which works well even with very large texts, while Euclidean distance can get confused by text length and high-dimensional data.
"""
"""
Build a small script:
Input two texts
Convert to TF-IDF
Calculate similarity
Explain the score
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# sample texts
text1 = "I love NLP"
text2 = "I enjoy NLP and text processing"

# Step 1: Converting to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([text1, text2])

# Step 2: Calculating cosine similarity
similarity_score = cosine_similarity(tfidf[0], tfidf[1])
print("Cosine Similarity:", similarity_score[0][0])

"""
Rank multiple documents based on similarity to a query
"""
query = "I love NLP"
docs = [
    "NLP is amazing",
    "I enjoy coding",
    "I love NLP and AI",
    "Python is fun"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([query] + docs)
similarities = cosine_similarity(tfidf[0], tfidf[1:])[0]
# Creating a list of indices
indices = list(range(len(docs)))

# Sorting indices based on similarity (highest first)
indices.sort(key=lambda i: similarities[i], reverse=True)

# Printing documents in ranked order
for i in indices:
    print(docs[i], "-> Similarity:", round(similarities[i], 2))


