# Multi-Keyword Search System

## Overview
This project implements a **Multi-Keyword Search System** using **TF-IDF, BM25, and BERT-based embeddings** for ranked information retrieval. It supports authentication, file uploads, and efficient searching across datasets.

## Features
- 🔐 **User Authentication:** Registration, login, and access control (admin grants search access).
- 📂 **CSV File Upload:** Users can upload CSV files for indexing.
- 🔍 **Multi-Keyword Search:** Uses a hybrid approach combining **TF-IDF**, **BM25**, and **BERT embeddings**.
- 📊 **Optimized Ranking:** Retrieves the most relevant results using top **TF-IDF & BM25 scores**.
- 🖥️ **Streamlit UI:** Simple and interactive user interface.
- 🚀 **Scalability:** Can handle large datasets with efficient retrieval.

## Technologies Used
- **Python** 🐍
- **Streamlit** 🎨 (Frontend UI)
- **Pandas** 📊 (Data handling)
- **Scikit-learn** 🤖 (TF-IDF Vectorization)
- **Rank-BM25** 📚 (BM25 Algorithm)
- **SentenceTransformers** (BERT Embeddings)
- **Pickle** 🏺 (Model Persistence)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Setup & Run
```bash
# Clone the repository
git clone https://github.com/yourusername/multi-keyword-search.git
cd multi-keyword-search

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Usage
1. **Register/Login**: Create an account and log in.
2. **Upload CSV**: Upload dataset files (must be in CSV format).
3. **Search**: Enter a search query to find relevant results.
4. **Admin Access**: Admin can grant search access to users.

## File Structure
```
📂 multi-keyword-search
│── 📜 app.py              # Main Streamlit app
│── 📜 requirements.txt    # Required dependencies
│── 📂 uploads             # Folder for uploaded files
│── 📜 README.md           # Project Documentation
```

## Contribution
Feel free to contribute! Fork the repo, create a new branch, and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions or issues, reach out via [GitHub Issues](https://github.com/yourusername/multi-keyword-search/issues).

