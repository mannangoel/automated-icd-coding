# src/ttppr/stage2_rag/retriever.py
import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer


class ICD10Retriever:

    def __init__(
        self,
        index_path: str | Path,
        meta_path: str | Path,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                "FAISS index or metadata pickle file missing from data/processed/"
            )

        print("Loading FAISS Index and Metadata lookup table...")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer(model_name)

    def retrieve_code_context(
        self, query_text: str, top_k: int = 3
    ) -> list[dict]:
        """Queries FAISS using a clinical text string or candidate code description."""
        query_vector = self.model.encode(
            [query_text], normalize_embeddings=True
        )
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["similarity_score"] = float(score)
                results.append(item)
        return results

    def get_rules_for_candidates(
        self, candidate_codes: list[str]
    ) -> list[dict]:
        """Direct lookup of structured Excludes rules for Stage 1 candidate codes."""
        matched_rules = []
        for item in self.metadata:
            if item["code"] in candidate_codes:
                matched_rules.append(item)
        return matched_rules