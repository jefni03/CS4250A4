from pymongo import MongoClient
import re
import math
from collections import defaultdict

# Connect to MongoDB and initialize the database
client = MongoClient("mongodb://localhost:27017/")  # Adjust this connection if necessary
db = client.inverted_index
index_collection = db.index
docs_collection = db.docs

# Document dataset for indexing
documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The patient reported nausea and dizziness caused by the medication.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported."
]

# Function to clean and break text into terms (tokens)
def process_text(text):
    """Lowercase the text, remove punctuation, and generate unigrams, bigrams, and trigrams."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = text.split()
    # Use list comprehension to create n-grams
    ngrams = words + [" ".join(words[i:i+n]) for n in (2, 3) for i in range(len(words) - n + 1)]
    return ngrams

# Reset collections in MongoDB
index_collection.drop()
docs_collection.drop()

# Insert documents into MongoDB with unique IDs
for doc_id, content in enumerate(documents, start=1):
    docs_collection.insert_one({"_id": doc_id, "content": content})

# Create the inverted index
vocab = {}
for doc_id, content in enumerate(documents, start=1):
    # Extract tokens from the document
    tokens = process_text(content)
    token_freq = defaultdict(int)
    for token in tokens:
        token_freq[token] += 1  # Count occurrences of each token

    # Update the term index
    for term, count in token_freq.items():
        if term not in vocab:
            vocab[term] = len(vocab) + 1  # Assign a unique position to each term
        term_id = vocab[term]
        tf = 1 + math.log(count)  # Calculate term frequency (TF)
        idf = math.log(len(documents) / sum(term in process_text(d) for d in documents))  # Calculate IDF
        tf_idf = tf * idf

        # Insert or update term data in MongoDB
        index_collection.update_one(
            {"_id": term_id},
            {
                "$set": {"term": term},
                "$addToSet": {"documents": {"doc_id": doc_id, "score": tf_idf}}
            },
            upsert=True
        )

# Queries to be executed
queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication"
]

# Function to calculate relevance scores for documents
def calculate_relevance(query):
    """Score each document based on matching terms using the inverted index."""
    tokens = process_text(query)
    scores = defaultdict(float)

    # Retrieve term data from the index for each token
    for token in tokens:
        term_data = index_collection.find_one({"term": token})
        if term_data:
            for doc_info in term_data["documents"]:
                scores[doc_info["doc_id"]] += doc_info["score"]

    # Sort documents by score in descending order
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Process each query and collect results
results = {}
for query_id, query in enumerate(queries, start=1):
    relevant_docs = calculate_relevance(query)
    results[f"Query {query_id}"] = [
        {"content": documents[doc_id - 1], "score": round(score, 3)}
        for doc_id, score in relevant_docs
    ]

# Print results for all queries
for query_key, docs in results.items():
    print(f"\nResults for {query_key}:")
    for doc in docs:
        print(f"Document: {doc['content']} | Relevance Score: {doc['score']}")
