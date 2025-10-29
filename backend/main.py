"""
Medical AI Chatbot Backend - FastAPI + Agno
A production-ready medical domain chatbot with RAG, session management, and LLM fallback
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.groq import Groq
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.db.sqlite import SqliteDb

import sqlite3

from fastapi import Request

import pytz


from dotenv import load_dotenv
import asyncio

from fastapi.middleware.cors import CORSMiddleware


# ===============================
# Load environment variables
# ===============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Please add it to your .env file.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print(f"Loaded GROQ_API_KEY: {GROQ_API_KEY[:4]}***")
print(f"Using Model: {OPENAI_MODEL}")

# ===============================
# Logging setup
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHROMA_PATH = "./medical_vectordb"
SQLITE_DB = "./medical_agent.db"

# ==========================
# FEEDBACK DATABASE SETUP
# ==========================
def init_db():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()


# ===============================
# Request / Response Models
# ===============================
class QueryRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    session_id: str
    user_id: str
    timestamp: str
    sources: Optional[list] = None
    model_used: str

# ===============================
# Medical Domain Validator
# ===============================
class MedicalDomainValidator:
    MEDICAL_KEYWORDS = [
    # General medical terms
    "disease", "symptom", "treatment", "medicine", "diagnosis", "health",
    "doctor", "patient", "medical", "clinical", "therapy", "drug", "condition",
    "pain", "fever", "infection", "cancer", "diabetes", "blood", "heart",

    # Body systems & anatomy
    "brain", "lungs", "kidney", "liver", "cough", "fatigue","stomach", "intestine", "skin", "eye",
    "ear", "nose", "throat", "spine", "bone", "muscle", "nerve", "heart rate",
    "blood pressure", "oxygen", "immune", "digestive", "respiratory", "circulatory",

    # Common illnesses & disorders
    "asthma", "allergy", "hypertension", "cholesterol", "stroke", "migraine",
    "ulcer", "thyroid", "arthritis", "depression", "anxiety", "flu", "cold",
    "pneumonia", "covid", "covid-19", "malaria", "tuberculosis", "infection",
    "anemia", "hepatitis", "obesity", "epilepsy", "autism", "dementia",

    # Medical procedures & diagnostics
    "surgery", "operation", "scan", "x-ray", "mri", "ct scan", "ultrasound",
    "biopsy", "test", "blood test", "ecg", "ekg", "endoscopy", "radiology",
    "pathology", "diagnostic", "lab", "examination", "screening",

    # Treatment & medications
    "antibiotic", "vaccine", "vaccination", "dose", "tablet", "capsule",
    "ointment", "injection", "therapy", "chemotherapy", "radiation",
    "rehabilitation", "prescription", "dosage", "supplement", "painkiller",
    "paracetamol", "ibuprofen", "antiviral", "insulin",

    # Healthcare & wellness
    "hospital", "clinic", "nurse", "pharmacy", "appointment", "consultation",
    "emergency", "first aid", "therapy session", "physiotherapy", "mental health",
    "nutrition", "diet", "exercise", "fitness", "wellness", "sleep", "stress",

    # Specific medical topics
    "cardiology", "neurology", "dermatology", "oncology", "psychiatry",
    "orthopedic", "gynecology", "pediatrics", "urology", "endocrinology",
    "ophthalmology", "dentistry", "immunology", "pathogen", "virus", "bacteria",

    # Miscellaneous
    "symptomatic", "diagnostic", "differential diagnosis", "prescribed",
    "side effects", "vitals", "chronic", "acute", "infection control",
    "inflammation", "wound", "fracture", "recovery", "rehab", "consult",
    "telemedicine", "telehealth"
]

    @staticmethod
    def is_medical_query(query: str) -> bool:
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in MedicalDomainValidator.MEDICAL_KEYWORDS)


# ===============================
# LLM Manager
# ===============================
class LLMManager:
    def __init__(self):
        self.primary_model = Groq(
            id=OPENAI_MODEL,
            api_key=GROQ_API_KEY
        )
        logger.info(f"Initialized primary model: {OPENAI_MODEL} (Groq)")

    def get_model(self):
        return self.primary_model


# ===============================
# Local Embedding Wrapper
# ===============================
from sentence_transformers import SentenceTransformer

class LocalEmbedderWrapper:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SentenceTransformer model: {model_name}")
        logger.info(f"Initialized local embedding model: {model_name}")

    def get_embedding(self, text):
        """
        Accepts a string or a list of strings and returns embedding(s) as list of floats.
        Compatible with ChromaDb.
        """
        # If single string, wrap it in a list
        single_input = False
        if isinstance(text, str):
            text = [text]
            single_input = True

        # Get embeddings as numpy array
        embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

        # Convert numpy array(s) to list(s)
        embeddings_list = embeddings.tolist()

        # If single input, return a single list
        if single_input:
            return embeddings_list[0]
        return embeddings_list

    # Optional: dummy method so batch handling works
    def enable_batch(self, batch_size=32):
        return self



# ===============================
# Embedding Manager
# ===============================
class EmbeddingManager:
    def __init__(self):
        self.embedder = LocalEmbedderWrapper()
        logger.info("Initialized local embedding model: all-MiniLM-L6-v2")

    def get_embedder(self):
        return self.embedder


# ===============================
# RAG Retriever
# ===============================
class RAGRetriever:
    def __init__(self, embedder):
        self.vector_db = ChromaDb(
            collection="medical_documents",
            path=CHROMA_PATH,
            persistent_client=True,
            embedder=embedder
        )

        self.knowledge = Knowledge(
            name="Medical Knowledge Base",
            description="Curated medical documents for RAG",
            vector_db=self.vector_db,
            max_results=5
        )
        logger.info("Initialized ChromaDB vector store")

    async def add_documents(self, path: str, metadata: Dict[str, Any] = None):
        try:
            if not os.path.exists(path):
                logger.warning(f"Document folder not found: {path}")
                return
            await self.knowledge.add_content_async(path=path, metadata=metadata)
            logger.info(f"Added documents from: {path}")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")


# ===============================
# Medical Agent
# ===============================
class MedicalAgent:
    def __init__(self, llm_manager: LLMManager, knowledge: Knowledge, db: SqliteDb):
        self.llm_manager = llm_manager
        self.validator = MedicalDomainValidator()
        self.agent = Agent(
            name="Medical Assistant",
            model=llm_manager.get_model(),
            knowledge=knowledge,
            db=db,
            add_history_to_context=True,
            num_history_runs=5,
            search_knowledge=True,
            instructions=[
                "You are a medical information assistant.",
                "Only answer questions related to healthcare, diseases, medicines, symptoms, treatments, or diagnosis.",
                "If asked about non-medical topics, politely decline.",
                "Always cite sources when providing medical information.",
                "Never provide definitive diagnoses - recommend consulting healthcare professionals."
            ],
            markdown=True
        )
        logger.info("Medical Agent initialized")

    async def process_query(self, query: str, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process an incoming user query. Handles simple greetings and domain filtering,
        then routes to the agent for medical queries.
        """
        message = query.lower().strip()

        # --- Step 1: Handle greetings ---
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if any(greet in message for greet in greetings):
            return {
                "response": "Hello! 👋 I'm Medi-Assist, your medical assistant. How can I help you today?",
                "session_id": session_id or "N/A",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sources": None,
                "model_used": "greeting"
            }

        # --- Step 2: Check if it's a medical query ---
        if not self.validator.is_medical_query(query):
            return {
                "response": "I'm designed to assist only with medical-related queries.",
                "session_id": session_id or "N/A",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sources": None,
                "model_used": "domain_filter"
            }

        # --- Step 3: Continue with normal medical processing ---
        try:
            response = self.agent.run(
                input=query,
                user_id=user_id,
                session_id=session_id or f"session_{user_id}_{datetime.utcnow().timestamp()}",
                stream=False
            )

            return {
                "response": response.content,
                "session_id": response.session_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "sources": getattr(response, 'references', None),
                "model_used": OPENAI_MODEL
            }

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            raise HTTPException(status_code=500, detail="Error processing query")


# ===============================
# FastAPI app + Lifespan
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Medical AI Chatbot Backend...")
    app.state.llm_manager = LLMManager()
    app.state.embedding_manager = EmbeddingManager()
    app.state.rag_retriever = RAGRetriever(app.state.embedding_manager.get_embedder())
    app.state.db = SqliteDb(db_file=SQLITE_DB)
    app.state.medical_agent = MedicalAgent(
        llm_manager=app.state.llm_manager,
        knowledge=app.state.rag_retriever.knowledge,
        db=app.state.db
    )

    await app.state.rag_retriever.add_documents("./medical_docs/")

    logger.info("Application startup complete")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="Medical AI Chatbot",
    description="RAG-powered medical domain chatbot with session management",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# Endpoints
# ===============================
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    result = await app.state.medical_agent.process_query(
        query=request.query,
        user_id=request.user_id,
        session_id=request.session_id
    )
    return QueryResponse(**result)


@app.get("/health")
async def health_check():
    ist = pytz.timezone("Asia/Kolkata")
    current_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    return {"status": "healthy", "timestamp": current_time}



@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    return {"user_id": user_id, "sessions": []}


@app.post("/feedback")
async def feedback(request: Request):
    data = await request.json()
    feedback_text = data.get("feedback")
    timestamp = data.get("timestamp", datetime.utcnow().isoformat())

    if not feedback_text:
        return {"error": "Feedback text is required"}

    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (feedback, timestamp) VALUES (?, ?)", (feedback_text, timestamp))
    conn.commit()
    conn.close()

    return {"message": "✅ Feedback stored successfully"}



@app.get("/admin/stats")
async def admin_stats():
    return {
        "total_queries": 0,
        "active_sessions": 0,
        "avg_response_time": 0,
        "message": "Admin dashboard coming soon"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
