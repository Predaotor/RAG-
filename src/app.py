"""Streamlit demo app for the RAG agent."""

import sys
from pathlib import Path

# Allow running as script directly (streamlit run src/app.py)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from src.config import CITATION, DATA_DIR, OPENAI_API_KEY, VECTORSTORE_PATH
from src.loader import load_documents, split_documents
from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore


def init_vectorstore() -> VectorStore:
    """Initialize or load the vector store."""
    vs = VectorStore()
    if vs.load():
        return vs

    # Build from documents
    docs = load_documents()
    if not docs:
        return vs

    chunks = split_documents(docs)
    vs.add_documents(chunks)
    vs.save()
    return vs


def main():
    st.set_page_config(
        page_title="RAG აგენტი - საგადასახადო და საბაჟო ჰაბი",
        page_icon="📋",
        layout="centered",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Georgian:wght@400;600;700&display=swap');
    .main { font-family: 'Noto Sans Georgian', sans-serif; }
    .title { color: #1a365d; font-size: 2rem; margin-bottom: 0.5rem; }
    .subtitle { color: #4a5568; font-size: 1rem; margin-bottom: 2rem; }
    .citation-box { 
        background: #edf2f7; 
        padding: 1rem; 
        border-radius: 8px; 
        border-left: 4px solid #2b6cb0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="title">📋 RAG აგენტი</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">საინფორმაციო და მეთოდოლოგიურ ჰაბი - საგადასახადო და საბაჟო ადმინისტრირების დოკუმენტები</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="citation-box">წყარო: {CITATION}</div>',
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.warning(
            "⚠️ **OPENAI_API_KEY** არ არის დაყენებული. გთხოვთ, შექმნათ `.env` ფაილი საქაღალდის ფესვში "
            "ან დააყენოთ გარემოს ცვლადი: `OPENAI_API_KEY=your-key`"
        )

    if "vectorstore" not in st.session_state:
        with st.spinner("ვექტორული მაღაზია იტვირთება..."):
            st.session_state.vectorstore = init_vectorstore()

    if "rag" not in st.session_state:
        st.session_state.rag = RAGPipeline(st.session_state.vectorstore)

    vectorstore = st.session_state.vectorstore
    rag = st.session_state.rag

    # Check if we have documents
    if len(vectorstore.documents) == 0:
        st.info(
            "📁 დოკუმენტები ჯერ არ არის ჩატვირთული. გთხოვთ, მოათავსოთ PDF, DOCX ან TXT ფაილები "
            f"`{DATA_DIR}` საქაღალდეში და დააჭიროთ ქვემოთ მოცემულ ღილაკს."
        )
        if st.button("🔄 ხელახლა ჩატვირთვა"):
            st.rerun()

        # Sample question for demo
        st.divider()
        st.markdown("### დემო კითხვა")
        st.markdown("დოკუმენტების ჩატვირთვის შემდეგ შეგიძლიათ კითხვების დასმა.")
    else:
        st.success(f"✅ ჩატვირთულია {len(vectorstore.documents)} დოკუმენტის ფრაგმენტი.")

        question = st.text_input(
            "კითხვა",
            placeholder="მაგალითად: როგორ უნდა შევავსო საგადასახადო დეკლარაცია?",
            label_visibility="collapsed",
        )

        if question:
            with st.spinner("პასუხი მზადდება..."):
                answer = rag.query(question)

            st.markdown("### პასუხი")
            st.markdown(answer)
            st.divider()

        if st.button("🔄 დოკუმენტების ხელახლა ჩატვირთვა"):
            # Clear and rebuild
            (VECTORSTORE_PATH / "index.faiss").unlink(missing_ok=True)
            (VECTORSTORE_PATH / "documents.pkl").unlink(missing_ok=True)
            del st.session_state["vectorstore"]
            del st.session_state["rag"]
            st.rerun()

    st.divider()
    st.markdown(
        "[საინფორმაციო და მეთოდოლოგიურ ჰაბი - infohub.rs.ge](https://infohub.rs.ge/ka)"
    )


if __name__ == "__main__":
    main()
