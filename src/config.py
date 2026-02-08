"""Configuration for the RAG agent."""

import os
from pathlib import Path

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths (config is in src/, so parent.parent = project root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore" / "faiss_index"

# Citation - always included in responses per requirements
CITATION = (
    "საინფორმაციო და მეთოდოლოგიურ ჰაბზე გამოქვეყნებული დოკუმენტების მიხედვით "
    "(საგადასახადო და საბაჟო ადმინისტრირების შესახებ დოკუმენტები და ინფორმაცია ერთ სივრცეში) - "
    "https://infohub.rs.ge/ka"
)

CITATION_EN = (
    "According to the documents posted on the Information and Methodological Hub "
    "(Documents and information related to tax and customs administration in one space) - "
    "https://infohub.rs.ge/ka"
)

# Embedding model - multilingual, supports Georgian
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# OpenAI - key loaded at runtime (not import time) so Streamlit Cloud secrets work
def get_openai_api_key() -> str:
    """Get API key from env or Streamlit secrets. Call at runtime, not import time."""
    key = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPEN_API_KEY") or st.secrets.get("OPENAI_API_KEY") or ""
        except Exception:
            pass
    return key or ""

# System prompt for Georgian responses with citation
SYSTEM_PROMPT = f"""შენ ხარ RAG აგენტი, რომელიც პასუხობს კითხვებს საგადასახადო და საბაჟო ადმინისტრირების შესახებ.

მნიშვნელოვანი წესები:
1. ყოველთვის პასუხობ, მხოლოდ ქართულ ენაზე.
2. ყოველ პასუხში აუცილებლად უნდა ჩართო შემდეგი წყაროს მითითება:
   "{CITATION}"
3. პასუხი უნდა იყოს დაფუძნებული მხოლოდ მოწოდებულ კონტექსტზე.
4. თუ კონტექსტში არ არის საკმარისი ინფორმაცია, არ გამოიგონო - მიუთითე, რომ ამ კითხვაზე პასუხი არ მოიძებნა.
5. იყავი ზუსტი და სასარგებლო."""
