# NLP Phase 3: Vectorization Tasks

This phase focuses on **converting text documents into numeric vectors** using Bag of Words (BoW) and TF-IDF, and comparing the results.

---

## Task 1: Manually Create Vocabulary & BoW Vectors for Small Documents

### Documents
1. Doc1: "I love NLP"  
2. Doc2: "NLP is amazing"  
3. Doc3: "I love coding"

### Step 1: Creating Vocabulary
- Vocabulary = all unique words from all documents:  
["I", "love", "NLP", "is", "amazing", "coding"]

### Step 2: Creating Bag of Words (BoW) Vectors
- Count how many times each word appears in each document  

| Document | I | love | NLP | is | amazing | coding |
|----------|---|------|-----|----|---------|--------|
| Doc1     | 1 | 1    | 1   | 0  | 0       | 0      |
| Doc2     | 0 | 0    | 1   | 1  | 1       | 0      |
| Doc3     | 1 | 1    | 0   | 0  | 0       | 1      |

> **Explanation:**  
> - Rows = Documents  
> - Columns = Vocabulary words  
> - Numbers = Count of the word in that document  

---

## Task 2: Explain in Words – TF, IDF, TF-IDF

### TF (Term Frequency)
TF measures the frequency of a word within a document. It is calculated as the ratio of the number of times a word occurs in a document to the total number of words in that document. 
- **Example:**  
  - Document: "I love NLP"  
  - Total words = 3  
  - TF("I") = 1/3 ≈ 0.33  
  - TF("love") = 1/3 ≈ 0.33  

### IDF (Inverse Document Frequency)
IDF measures the rarity of a term across a collection of documents. It is calculated as the logarithm of the ratio of the total number of documents to the number of documents containing the term

- **Example:**  
  - Total documents = 3  
  - Word “I” appears in 2 documents → IDF("I") = log(3/2) ≈ 0.18 (low, because common)  
  

### TF-IDF
 TF-IDF = **TF × IDF**  
- **Why it exists:**  
  - Simple Bag of Words only counts words → common words dominate → not useful  
  - TF-IDF reduces weight of common words and increases weight of rare, meaningful words  
  - TF-IDF exists to highlight important words in a document by combining **frequency** and **rarity**, making text representation more meaningful for NLP and machine learning tasks.
---

## Task 3: Convert Text to BoW, Convert Text into TF-IDF, Compare Results

The solution for this task has been implemented in the Python file:

**`main.py`**

