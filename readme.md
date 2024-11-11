# README for NLP-Based Search System using BERT, ChromaDB & SQuAD

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model & Tokenization](#model--tokenization)
5. [Embedding Creation](#embedding-creation)
6. [Database Storage](#database-storage)
7. [Search System](#search-system)
8. [Evaluation](#evaluation)
9. [Conclusion](#conclusion)

## Overview

This repository implements a Question Answering (QA) system using **BERT** embeddings, the **SQuAD** dataset, and **ChromaDB** for storage and retrieval of context passages. The main goal is to embed textual data (context from QA pairs) and store these embeddings in **ChromaDB**. Given a query (a question), the system searches the database for the most relevant passages and retrieves them.

The system also includes an evaluation function that compares true answers with the retrieved passages to calculate precision.

## Installation

To install the necessary dependencies for this project, run the following command:

```bash
#!pip install datasets transformers chromadb
!pip install -r requirements.txt
```

This installs the following packages:
- `datasets` for working with common NLP datasets like SQuAD.
- `transformers` for loading and using pre-trained models such as BERT.
- `chromadb` for managing document storage and query functionalities via embeddings.

## Dataset

The dataset being used is the **Stanford Question Answering Dataset (SQuAD)**. In this project, we only use a small subset of 1000 questions and contexts for faster processing.

**Relevant Code Parts:**

```python
from datasets import load_dataset
dataset = load_dataset("rajpurkar/squad")
train_dataset = dataset["train"].select(range(1000))
```

1. `load_dataset`: Loads the SQuAD dataset.
2. We are specifically using the **train** split and selecting a smaller subset (`1000` examples) for efficiency and testing.

## Model & Tokenization

We use BERT (Bidirectional Encoder Representations from Transformers) from the `Hugging Face` library to create embeddings from contexts. These embeddings will later be stored in the database.

**Relevant Code Parts:**

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
```

- `AutoTokenizer`: Tokenizes the input text, making it digestible by the BERT model.
- `AutoModel`: Loads the pre-trained BERT model (`bert-base-uncased`).

### Function: `embed_text`
This function takes a chunk of text (usually the context) and generates a **mean-pooled** embedding using BERT. The output is a `768-dimensional` vector representation of the input.

```python
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
```

- The text is tokenized and passed through BERT.
- The embeddings are **mean-pooled** for simplicity (`mean(dim=1)`).

## Database Storage

We use **ChromaDB** to store the context embeddings along with their respective metadata (such as the context and question ids). ChromaDB allows us to add documents in **batches** as well as perform fast, indexed similarity searches.

### Code Overview:

```python
client = chromadb.Client()
collection = client.create_collection(name="squad_contexts")
```

1. `ChromaDB.Client`: Initializes the ChromaDB client.
2. `create_collection`: Creates a collection named `squad_contexts`, which will store embeddings and their metadata.

### Batch Processing

We embed the contexts in batches instead of doing them one by one for efficiency:

```python
batch_size = 32
len_batches = len(train_dataset) // batch_size + 1
for i, example in enumerate(train_dataset):
    context = example["context"]
    context_embedding = embed_text(context)
```

- Embeddings are processed in batches of 32 cases per iteration.

- The embeddings, their associated context, metadata (e.g., context ID), and IDs are collected. After processing a batch, the function adds the batch to the **ChromaDB** collection:

```python
collection.add(
    embeddings=batch_embeddings,
    documents=batch_contexts,
    metadatas=batch_metadatas,
    ids=batch_ids
)
```

## Search System

With the embeddings stored, we can now perform a **similarity search**. Given a query (a question), the model generates an embedding for the query and retrieves the top N most similar context embeddings from the dataset using Chroma.

Example:

```python
def search_query(query, n_results=1):
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results
```

- `embed_text(query)` creates the query embedding.
- The `collection.query` function retrieves the closest `n_results` context passages based on cosine similarity of the embeddings.

**An example search:**
```python
query = "What is the capital of France?"
results = search_query(query)
```

- For query `"What is the capital of France?"`, the system finds the best matching context passages.

## Evaluation

The system also includes an evaluation metric that measures **precision** by comparing the true answer to which passages were retrieved in response to a given question.

### Precision Calculation:

For every question in the dataset:
1. Use BERT to embed the **question**.
2. Perform a search on **ChromaDB** to retrieve context passages.
3. Check if the **true answer** to the question appears in any of the retrieved passages.

Each correct retrieval (true answer in retrieved passage) contributes to the **total precision** score.

**Code for Precision Calculation:**

```python
def evaluate_search_engine():
    total_precision = 0
    n_questions = len(train_dataset)

    for example in train_dataset:
        question = example["question"]
        true_answer = example["answers"]["text"][0]
        results = collection.query(query_embeddings=[embed_text(question)], n_results=20)
        retrieved_answers = results['documents'][0]
        relevant = any(true_answer in passage for passage in retrieved_answers)
        total_precision += int(relevant)
        
    precision = total_precision / n_questions
    return precision
```

This function evaluates over all 1000 questions we loaded, computes the precision, and returns it.

## Conclusion

This repository implements a simple yet powerful **question-answering system** based on:
  - **BERT embeddings** for understanding the semantic similarity between questions and contexts.
  - **ChromaDB** to store and retrieve contexts based on similarity.
  - **Evaluation** of the search engine using precision for binary relevance check.

### Final Output

The following code prints the final precision of the model on the subset of the SQuAD dataset:

```python
precision = evaluate_search_engine()
print(f"Precision: {precision:.4f}")
```

This thorough pipeline provides a strong framework for experimenting with document storage and retrieval via pre-trained language model embeddings such as BERT.
