"""Loads policy + limit data and provides FAISS-based semantic retrieval over the policy."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class PolicyStore:
    def __init__(self, data_dir: Path = DATA_DIR, use_vector_store: bool = True):
        self.data_dir = data_dir
        self.policy_path = data_dir / "travel_policy.md"
        self.policy_text = self.policy_path.read_text(encoding="utf-8")
        self.limits = json.loads((data_dir / "limits.json").read_text(encoding="utf-8"))
        self.history = json.loads((data_dir / "claims_history.json").read_text(encoding="utf-8"))
        self._documents = self._load_policy_documents()
        self._vstore = self._build_vector_store() if use_vector_store else None
        self.retriever_kind = "faiss" if self._vstore is not None else "keyword"

    def _load_policy_documents(self) -> list:
        """Load the policy with a LangChain loader and split it into per-section chunks."""
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        raw = TextLoader(str(self.policy_path), encoding="utf-8").load()
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title"), ("##", "section")],
            strip_headers=False,
        )
        docs = []
        for doc in raw:
            for chunk in splitter.split_text(doc.page_content):
                heading = chunk.metadata.get("section") or chunk.metadata.get("title") or "Overview"
                chunk.metadata["heading"] = heading
                docs.append(chunk)
        return docs

    def _build_vector_store(self):
        """Index the loaded policy chunks in a FAISS vector store with open-source embeddings."""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            return FAISS.from_documents(self._documents, embeddings)
        except Exception as exc:  # offline / missing deps -> keyword fallback
            print(f"[policy_store] FAISS unavailable ({exc}); using keyword retrieval.")
            return None

    def search(self, query: str, top_k: int = 2) -> str:
        """Return the policy sections most semantically relevant to `query`.

        Uses the FAISS vector store when available, else a keyword fallback.
        """
        if self._vstore is not None:
            hits = self._vstore.similarity_search(query, k=top_k)
            return "\n\n".join(h.page_content.strip() for h in hits)
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 2) -> str:
        """Fallback retriever: simple keyword overlap over the loaded policy chunks."""
        terms = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for doc in self._documents:
            score = sum(doc.page_content.lower().count(t) for t in terms)
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return "No specific policy section matched; see general policy."
        return "\n\n".join(doc.page_content.strip() for _, doc in scored[:top_k])

    # --- structured accessors used by deterministic tools ---

    @property
    def eligible_categories(self) -> List[str]:
        return self.limits["eligible_categories"]

    @property
    def non_reimbursable_categories(self) -> List[str]:
        return self.limits["non_reimbursable_categories"]

    @property
    def category_limits(self) -> Dict[str, dict]:
        return self.limits["category_limits"]

    @property
    def receipt_required_above(self) -> float:
        return self.limits["receipt_required_above"]

    @property
    def submission_window_days(self) -> int:
        return self.limits["submission_window_days"]

    @property
    def director_threshold(self) -> float:
        return self.limits["director_threshold"]

    def required_approval(self, amount: float) -> str:
        for tier in self.limits["approval_matrix"]:
            if tier["max_amount"] is None or amount <= tier["max_amount"]:
                return tier["approval"]
        return "Director approval (Manual Review)"
