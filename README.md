# Intellignt_RAG_Chatbot
# 🧠 Intelligent Topic-Gated RAG Chatbot

A smart Retrieval-Augmented Generation (RAG) system that uses Machine Learning to classify user intent before retrieving information.
This "Topic-Gating" approach ensures the AI searches only the relevant knowledge base, reducing hallucinations and improving accuracy.

---

## 🚀 Features

* **ML-Powered Routing:** A trained Logistic Regression model classifies user queries into topics (e.g., Tech, Biology, Comedy) *before* searching.
* **Targeted RAG:** Maintains separate vector stores for different domains to ensure context retrieval is highly specific.
* **Generative AI:** Uses **Meta Llama 3** (via Groq) to generate human-like, context-aware responses.
* **Open Source Embeddings:** Utilizes HuggingFace's `all-MiniLM-L6-v2` for efficient and free text embeddings.
* **Local Vector Storage:** Uses FAISS (Facebook AI Similarity Search) for fast, local vector storage.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** LangChain
* **Machine Learning:** Scikit-Learn (TF-IDF Vectorization + Logistic Regression)
* **LLM Provider:** Groq (Llama-3.1-8b-instant)
* **Vector Database:** FAISS
* **Embeddings:** HuggingFace (`sentence-transformers`)

---

## 📂 Project Structure

```bash
├── documents/               # Raw knowledge base text files
│   ├── tech/                # Python programming docs
│   ├── biology/             # Snake biology docs
│   └── comedy/              # Monty Python docs
├── vector_stores/           # Generated FAISS indexes (created by script)
├── app.py                   # Main application (Run this to chat)
├── train_router.py          # Script to train the ML classifier
├── create_vector_stores.py  # Script to generate vector embeddings
├── queries.csv              # Training data for the ML model
├── requirements.txt         # Python dependencies
└── .env                     # API keys (Git ignored)
```
How to Run
This system is built in 3 modular steps:

Step 1: Train the Router
Train the Machine Learning model to recognize topics based on the data in queries.csv.

terminal

python train_router.py
Output: Generates router_model.joblib

Step 2: Build the Knowledge Base
Read documents from the documents/ folder and create the FAISS vector stores.

terminal

python create_vector_stores.py
Output: Populates the vector_stores/ directory.

Step 3: Run the Chatbot
Start the CLI application to interact with the AI.

terminal

python app.py
