# AI Tax Advisor Assistant for Business Owners

A Retrieval-Augmented Generation (RAG) based AI assistant that answers Pakistani tax questions using official Federal Board of Revenue (FBR) documents.

The system retrieves relevant chunks from indexed FBR documents and generates plain-English answers with citations. If the answer is not available in the indexed corpus, the assistant refuses gracefully.

---

## Project Goal

Pakistani tax rules are spread across many Acts, Ordinances, Rules, and official PDF documents. These documents are difficult for business owners to search and understand.

This project solves that problem by building an AI assistant that can:

- Accept tax questions through a Streamlit chat interface
- Retrieve relevant chunks from FBR documents
- Generate simple English answers
- Cite the source document, section/rule where visible, and year
- Refuse when the answer is not available in the indexed corpus
- Stream answers in real time
- Maintain limited chat history using a trimming strategy

---

## Features

- FBR document downloader
- PDF text extraction using `pypdf`
- Recursive paragraph-based chunking
- Multilingual chunk support for English and Urdu
- Local embeddings using Sentence Transformers
- ChromaDB vector store
- Groq API for answer generation
- Streamlit chat interface
- Citation-aware answer generation
- Graceful refusal for out-of-corpus questions
- Chat history trimming with a 1000-token limit

---

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Groq API
- pypdf
- BeautifulSoup
- Requests
- python-dotenv

---

## Project Structure

```text
ai-tax-advisor-fbr/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw_pdfs/
│   ├── extracted_text/
│   └── processed/
│
├── src/
│   ├── download_fbr_docs.py
│   ├── extract_pdf_text.py
│   ├── chunk_documents.py
│   ├── build_vectorstore.py
│   ├── retriever.py
│   └── rag_chain.py
│
├── vectorstore/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md