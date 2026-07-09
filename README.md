# AI Tax Advisor Assistant for Pakistani Business Owners

This project is a Retrieval-Augmented Generation (RAG) assistant that answers Pakistani tax questions using only official Federal Board of Revenue (FBR) documents.

## Goal

The assistant should:

- Download official FBR tax documents
- Extract and index their text
- Retrieve relevant document chunks for a user question
- Generate a plain-English answer using an LLM
- Cite the exact section, Act, and year where possible
- Refuse gracefully when the answer is not found in the indexed corpus

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- xAI/Grok API
- FBR official documents

## Project Status

Under development.