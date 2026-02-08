"""RAG pipeline for question answering in Georgian with source citation."""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import (
    CITATION,
    get_openai_api_key,
    SYSTEM_PROMPT,
    TOP_K_RETRIEVAL,
)
from .vectorstore import VectorStore


RAG_PROMPT = """{system_prompt}

კონტექსტი (წყარო დოკუმენტებიდან):
---
{context}
---

კითხვა: {question}

პასუხი (ქართულ ენაზე, აუცილებლად ჩართე წყაროს მითითება):"""


class RAGPipeline:
    """RAG pipeline that answers in Georgian and always cites the source."""

    def __init__(
        self,
        vectorstore: VectorStore,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.vectorstore = vectorstore
        self.api_key = api_key or get_openai_api_key()
        self.model_name = model

        self.llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=0.1,
        ) if self.api_key else None

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "კონტექსტი:\n{context}\n\nკითხვა: {question}\n\nპასუხი:"),
        ])

    def _format_context(self, documents: list) -> str:
        """Format retrieved documents as context."""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[{i}] {doc.page_content}")
        return "\n\n".join(parts)

    def _ensure_citation(self, response: str) -> str:
        """Ensure the citation is present in the response."""
        if CITATION not in response:
            response = response.rstrip()
            if not response.endswith("."):
                response += "."
            response += f"\n\n{CITATION}"
        return response

    def query(self, question: str, k: int | None = None) -> str:
        """
        Answer a question in Georgian, always citing the source.
        """
        k = k or TOP_K_RETRIEVAL
        docs = self.vectorstore.similarity_search(question, k=k)

        if not docs:
            return (
                "ამ კითხვაზე საკმარისი ინფორმაცია არ მოიძებნა მოწოდებულ დოკუმენტებში. "
                f"გთხოვთ, დარწმუნდით, რომ დოკუმენტები ჩატვირთულია. {CITATION}"
            )

        context = self._format_context(docs)

        if self.llm is None:
            # Fallback when no API key: return context with citation
            return (
                f"მოძიებული ინფორმაცია:\n\n{context}\n\n"
                f"{CITATION}\n\n"
                "(შენიშვნა: OpenAI API გასაღების დასაყენებლად გამოიყენეთ OPENAI_API_KEY ცვლადი.)"
            )

        chain = self.prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return self._ensure_citation(response)
