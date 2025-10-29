# 🩺 Medi-Assist: AI-Powered Medical Assistant

Medi-Assist is an intelligent healthcare assistant built with **FastAPI**, **Streamlit**, and **Agno**.  
It leverages **Retrieval-Augmented Generation (RAG)** and **local embeddings** to analyze medical documents,  
answer queries, and collect user feedback — all in a simple and warm-themed interface. 🌤️

---

## 🚀 Features

- 🧠 **AI-Powered Query Response** — Users can ask medical or health-related questions.
- 📄 **Document Ingestion** — Automatically ingests and embeds medical PDFs or text files.
- 💬 **Interactive Chat Interface** — Built with Streamlit, featuring warm color themes.
- 🗂️ **Local Vector Database** — Uses Chroma for efficient similarity search.
- 💾 **Feedback Storage** — Collects user feedback using an SQLite database (`feedback.db`).
- ⚙️ **Modular Backend** — Easy to extend or modify for new features or models.

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Embeddings | Agno + LocalEmbedderWrapper |
| Vector Database | Chroma (SQLite) |
| Feedback Storage | SQLite |
| Deployment | Render (Backend) + Streamlit Cloud (Frontend) |

---

## 🧩 Project Structure

Medi-Assist/
│
├── main.py # FastAPI backend
├── requirements.txt
├── .env # Environment variables (ignored in Git)
├── feedback.db # Stores user feedback (local only)
├── medical_agent.db # Stores agent memory
├── medical_vectordb/ # Chroma vector database
│ └── chroma.sqlite3
├── medical_docs/ # Folder for medical knowledge base docs
│ ├── file1.pdf
│ ├── file2.txt
│ └── ...
└── ui/
└── streamlit_app.py # Streamlit UI (Medi-Assist)


---

## ⚙️ Backend Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/health` | `GET` | Returns API health status |
| `/query` | `POST` | Takes a query message and returns AI response |
| `/feedback` | `POST` | Stores user feedback in SQLite DB |
| `/upload_docs` | `POST` | Adds new documents to the knowledge base |
| `/list_docs` | `GET` | Lists all available medical documents |
| `/delete_doc/{name}` | `DELETE` | Deletes a specific document |

---

## 💻 Streamlit UI

The **Streamlit UI** (`ui/streamlit_app.py`) connects to the FastAPI backend and allows:
- Entering health queries
- Viewing AI responses
- Uploading new documents
- Providing feedback

### 🧡 UI Highlights
- Title: **Medi-Assist**
- Warm theme: soft oranges and browns (`#fff6f0`, `#ffb47a`)
- Interactive response boxes
- Sidebar for document management and feedback

---

## 🧰 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Medi-Assist.git
cd Medi-Assist

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate     # on Windows
source venv/bin/activate  # on macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Start the Backend (FastAPI)
uvicorn main:app --reload


Backend will run at: http://localhost:8000

5️⃣ Start the Frontend (Streamlit)
streamlit run ui/streamlit_app.py


Frontend will run at: http://localhost:8501

👨‍💻 Author

Rahul Mendon
AI & Data Science Engineer | Passionate about intelligent healthcare solutions
📫 LinkedIn :- https://www.linkedin.com/in/rahul-mendon-65b33b257/