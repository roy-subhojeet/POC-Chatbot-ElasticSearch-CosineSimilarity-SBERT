# Reddit Information Retrieval Chatbot

This repository contains three distinct approaches for indexing and semantically searching through a collection of questions and answers sourced from Reddit.

## Approach 1: Chatbot on Elasticsearch

### Dependencies

- `elasticsearch`
- `json`

### Description

This approach uses Elasticsearch to index and search a corpus of questions. The code snippet reads questions and answers from a JSON file, creates embeddings for each question using Sentence Transformers, and then indexes these in Elasticsearch. Users can interactively search the indexed questions.

### Usage

1. Ensure Elasticsearch is running locally on port 9200.
2. Place the `result_politics.json` file containing the questions and answers in the same directory as the script.
3. Run the script and follow the on-screen instructions.

## Approach 2: Chatbot with Sentence Transformers

### Dependencies

- `sentence_transformers`
- `json`
- `torch`

### Description

This uses Sentence Transformers to create embeddings for questions and answers. The script reads data from a JSON file, pre-computes the embeddings for all the passages, and then interactively responds to user queries by finding the most similar passage in memory.

### Usage

1. Place the `result_politics.json` file containing the questions and answers in the same directory as the script.
2. Run the script and follow the on-screen instructions.

## Approach 3: Chatbot with TF-IDF Vectorization

### Dependencies

- `json`
- `nltk`
- `sklearn`

### Description

This approach applies TF-IDF vectorization and cosine similarity to create a chatbot. The script reads questions and answers from a JSON file and uses TF-IDF to represent the text. It then calculates cosine similarity scores to find the most similar answers to user queries.

### Usage

1. Place the `result_politics.json` file containing the questions and answers in the same directory as the script.
2. Run the script and follow the on-screen instructions.

## License and Academic Integrity Statement

This project is licensed under the MIT License.

### Academic Integrity Statement

By using this code, you agree to abide by the principles of academic integrity. Utilizing this code in academic projects or coursework without proper attribution may constitute plagiarism or other violations of academic integrity policies.

It is your responsibility to understand and comply with your institution's academic integrity guidelines. If you intend to use this code as part of an academic assignment or research, make sure to properly cite it, and consult with your instructor, supervisor, or appropriate academic staff to ensure that you are in compliance with all relevant policies and regulations.

