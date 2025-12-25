# 📚 LightRAG + Gemini (Dockerized RAG)

A lightweight **Retrieval-Augmented Generation (RAG)** project using **LightRAG (HKU)** and **Google Gemini** models.  
It ingests large text files, generates embeddings, and answers questions using multiple retrieval strategies.

---

## ✨ Features

- 🔍 Retrieval modes: **Naive, Local, Global, Hybrid**
- 🤖 LLM: `gemini-2.0-flash`
- 🧠 Embeddings: `text-embedding-004`
- ⚡ Async-first design
- 📦 Docker support
- 💾 Persistent vector storage

---

## 📁 Project Structure

```

app.py # Async RAG pipeline
demo.py # Minimal demo
Dockerfile
requirements.txt
rag_storage/ # Auto-created storage

```

---

## 🔑 Setup

```bash
export GEMINI_API_KEY="your_api_key_here"
pip install -r requirements.txt
```

---

## ▶️ Run

### Full Async App

```bash
python app.py
```
---

## 🧠 Models Used

| Purpose   | Model              |
| --------- | ------------------ |
| LLM       | gemini-2.0-flash   |
| Embedding | text-embedding-004 |

---

## 📜 License

MIT

---

⭐ Star the repo if you find it useful!

