# Converting text into Bow
# from sklearn importing countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn importing TfidVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# sample text corpus
corpus={
    "I love NLP and Machine Learning.",
    "Machine Learning is amazing",
    "I love learning new things."
}
# creating CountVectorizer object
# CountVectorizer converts text to numbers
vectorizer=CountVectorizer()
# Fit and transform corpus, fit creates the vocabulary and transform converts document to number array
bow_matrix=vectorizer.fit_transform(corpus)
# Get feature names (unique words in vocabulary)
print("Vocabulary:",vectorizer.get_feature_names_out())
# Printing sparse matrix
print("Bow Representation:\n",bow_matrix)
# converting sparse matrix to 2D array
print("Bow Representation:\n",bow_matrix.toarray())


# Converting text into TF-IDF
# from sklearn importing TfidVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# using same sample text corpus
# creating object
tfidf_vectorizer=TfidfVectorizer()
# Fit the vectorizer on the corpus and transform documents into TF-IDF numeric vectors
tfidf_matrix=tfidf_vectorizer.fit_transform(corpus)
# Get feature names (unique words in vocabulary)
print("TF-IDF Vocabulary:",tfidf_vectorizer.get_feature_names_out())
# Printing sparse matrix
print("TF-IDF Representation:\n",tfidf_matrix)
# converting sparse matrix to 2D array
print("TF-IDF Representation:\n",tfidf_matrix.toarray())