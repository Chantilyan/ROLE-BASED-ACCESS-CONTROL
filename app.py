import os
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Multi-Keyword Search", layout="wide")
st.title("🔍 Multi-Keyword Search System")

MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 **User Authentication Storage**
if "users" not in st.session_state:
    st.session_state.users = {"admin": hashlib.sha256("admin123".encode()).hexdigest()}
if "access" not in st.session_state:
    st.session_state.access = {"admin": True}  # Admin has full access
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# 🔹 **Data Storage for TF-IDF & BM25**
df_dict, tfidf_vectorizer_dict, tfidf_matrix_dict, bm25_dict = {}, {}, {}, {}

# 🔹 **Helper Function for Preprocessing**
def clean_text(text):
    """Preprocess text while keeping symbols like '&', ',' and '.'."""
    return text.lower().strip()

# 🔹 **Process CSV File**
def process_csv(filepath, filename):
    """Preprocess CSV file and create TF-IDF & BM25 models."""
    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    df.fillna("", inplace=True)
    
    df = df.applymap(str).applymap(clean_text)
    df["combined"] = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    # 🔹 **Load or Compute TF-IDF & BM25**
    tfidf_model_path = f"{filename}_tfidf.pkl"
    bm25_model_path = f"{filename}_bm25.pkl"

    if os.path.exists(tfidf_model_path) and os.path.exists(bm25_model_path):
        with open(tfidf_model_path, "rb") as f:
            tfidf_vectorizer, tfidf_matrix = pickle.load(f)
        with open(bm25_model_path, "rb") as f:
            bm25 = pickle.load(f)
    else:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined"])
        with open(tfidf_model_path, "wb") as f:
            pickle.dump((tfidf_vectorizer, tfidf_matrix), f)

        tokenized_corpus = [doc.split() for doc in df["combined"]]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(bm25_model_path, "wb") as f:
            pickle.dump(bm25, f)

    df_dict[filename] = df
    tfidf_vectorizer_dict[filename] = tfidf_vectorizer
    tfidf_matrix_dict[filename] = tfidf_matrix
    bm25_dict[filename] = bm25

# 🔹 **User Authentication**
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user():
    st.sidebar.subheader("📝 Register User")
    new_user = st.sidebar.text_input("Username", key="register_username")
    new_pass = st.sidebar.text_input("Password", type="password", key="register_password")
    
    if st.sidebar.button("Register", key="register_button"):
        if new_user in st.session_state.users:
            st.sidebar.error("❌ Username already exists!")
        else:
            st.session_state.users[new_user] = hash_password(new_pass)
            st.session_state.access[new_user] = False  # No search access by default
            st.sidebar.success("✅ User registered! Admin must grant access.")

def login_user():
    st.sidebar.subheader("🔑 Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    
    if st.sidebar.button("Login", key="login_button"):
        if username in st.session_state.users and st.session_state.users[username] == hash_password(password):
            st.session_state.current_user = username
            st.sidebar.success(f"✅ Logged in as {username}")
        else:
            st.sidebar.error("❌ Invalid credentials!")

def logout_user():
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.current_user = None
        st.sidebar.success("✅ Logged out!")

# 🔹 **Admin Grants Access**
def grant_access():
    if st.session_state.current_user == "admin":
        st.sidebar.subheader("🔑 Grant Access")
        user_to_grant = st.sidebar.selectbox("Select User", [u for u in st.session_state.users if u != "admin"], key="grant_access_user")
        
        if st.sidebar.button("Grant Search Access", key="grant_access_button"):
            st.session_state.access[user_to_grant] = True
            st.sidebar.success(f"✅ {user_to_grant} can now search.")

# 🔹 **File Uploading**
st.sidebar.header("📂 Upload CSV Files")
uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        process_csv(filepath, uploaded_file.name)
    st.sidebar.success("✅ Files Processed!")

# 🔹 **Search Query Function**
def search_query(query):
    """Retrieve relevant rows based on query using TF-IDF & BM25 ranking."""
    query = clean_text(query)
    results = []

    for filename, df in df_dict.items():
        tfidf_vectorizer = tfidf_vectorizer_dict[filename]
        tfidf_matrix = tfidf_matrix_dict[filename]
        bm25 = bm25_dict[filename]

        # 🔹 **TF-IDF Search (Top 10%)**
        query_tfidf = tfidf_vectorizer.transform([query])
        tfidf_scores = np.ravel(tfidf_matrix.dot(query_tfidf.T).toarray())
        tfidf_threshold = np.percentile(tfidf_scores, 90)
        tfidf_top_indices = np.where(tfidf_scores >= tfidf_threshold)[0]

        # 🔹 **BM25 Search (Top 10%)**
        bm25_scores = bm25.get_scores(query.split())
        bm25_threshold = np.percentile(bm25_scores, 90)
        bm25_top_indices = np.where(bm25_scores >= bm25_threshold)[0]

        # 🔹 **Final Combined Results (Top 25)**
        final_indices = sorted(set(tfidf_top_indices).union(set(bm25_top_indices)), key=lambda idx: -tfidf_scores[idx])[:25]

        for idx in final_indices:
            result = df.iloc[idx].to_dict()
            result.pop("combined", None)  # Remove 'combined' column
            result["source_file"] = filename
            result["score"] = tfidf_scores[idx]
            results.append(result)

    return results if results else [{"Message": "🔎 No relevant results found."}]

# 🔹 **Search Section**
st.header("🔎 Search Query")
if st.session_state.current_user and st.session_state.access.get(st.session_state.current_user, False):
    query = st.text_input("Enter Search Query:", key="search_query")
    
    if query:
        search_results = search_query(query)
        st.dataframe(pd.DataFrame(search_results))
else:
    st.warning("🚫 You don't have access to search. Ask admin for permission.")

# Display authentication options
register_user()
login_user()
logout_user()
grant_access()
