import streamlit as st
import requests
from datetime import datetime
import uuid

# ===========================
# CONFIG
# ===========================
API_BASE_URL = "http://127.0.0.1:8000"  # Change when deployed
st.set_page_config(page_title="Medi-Assist", page_icon="💊", layout="wide")

# ===========================
# THEME STYLING
# ===========================
st.markdown("""
<style>
    .stApp { background-color: #fff6f0; color: #4a2c2a; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #b3541e; }
    div[data-testid="stSidebar"] { background-color: #ffe6d5; }
    .stButton>button {
        background-color: #ffb47a; color: white; border: none;
        border-radius: 10px; padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #e6904e; }
    .response-box {
        background-color: #fff3e3; padding: 15px; border-radius: 10px; margin-top: 10px;
    }
   div[data-testid="stTextArea"] label p {
    color: grey !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# SIDEBAR
# ===========================
st.sidebar.title("💊 Medi-Assist")
page = st.sidebar.radio("Navigate", ["💬 Chat", "📊 Dashboard", "🗒 Feedback"])

# ===========================
# PAGE 1: CHAT
# ===========================

if page == "💬 Chat":
    st.title("🩺 Ask Medi-Assist")
    st.caption("An AI-powered medical assistant trained with healthcare knowledge.")
    
    # Automatically generate a unique user ID per Streamlit session
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

    user_id = st.session_state.user_id
    session_id = None  # Let backend handle it automatically


    query = st.text_area("💬 Ask your medical question:")

    if st.button("Ask"):
        if query.strip():
            with st.spinner("Processing your query..."):
                try:
                    payload = {
                        "query": query,
                        "user_id": user_id,
                        "session_id": session_id if session_id else None
                    }
                    response = requests.post(f"{API_BASE_URL}/query", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(
                            f"<div class='response-box'><b>🧠 Response:</b><br>{data['response']}</div>",
                            unsafe_allow_html=True
                        )
                        st.info(f"Model used: {data['model_used']}")
                        if data.get("sources"):
                            st.write("📚 **Sources:**", data["sources"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        else:
            st.warning("Please type a question before submitting.")

# ===========================
# PAGE 2: DASHBOARD
# ===========================
elif page == "📊 Dashboard":
    st.title("📊 System Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Health Check")
        try:
            health = requests.get(f"{API_BASE_URL}/health").json()
            st.markdown(
    f"<p style='color:black;'>Server is {health['status']} (as of {health['timestamp']})</p>",
    unsafe_allow_html=True
)

        except Exception:
            st.error("Server not reachable.")

    with col2:
        st.subheader("📈 Admin Stats")
        try:
            stats = requests.get(f"{API_BASE_URL}/admin/stats").json()
            st.json(stats)
        except Exception:
            st.error("Could not fetch admin stats.")

# ===========================
# PAGE 3: FEEDBACK
# ===========================
elif page == "🗒 Feedback":
    st.title("🗒 Share Your Feedback")
    st.caption("We appreciate your thoughts to make Medi-Assist better!")

    feedback_text = st.text_area("💭 Write your feedback:")
    if st.button("Submit"):
        if feedback_text.strip():
            try:
                payload = {"feedback": feedback_text, "timestamp": datetime.utcnow().isoformat()}
                res = requests.post(f"{API_BASE_URL}/feedback", json=payload)
                if res.status_code == 200:
                    st.markdown("<p style='color:black;'>✅ Feedback submitted successfully. Thank you!</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color:red;'>Failed to submit feedback.</p>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("<p style='color:red;'>Please write something before submitting.</p>", unsafe_allow_html=True)

