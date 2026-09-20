# src/ttppr/stage2_rag/index_builder.py
import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_icd10_vector_store(
    json_path: str | Path,
    output_dir: str | Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    """Embeds parsed ICD-10 code rules and stores them in a FAISS index."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON file not found at: {json_path}")

    print(f"Loading parsed ICD-10 rules from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    documents = []
    metadata = []

    for item in records:
        code = item["code"]
        desc = item["description"]
        ex1 = "; ".join(item.get("excludes1", []))
        ex2 = "; ".join(item.get("excludes2", []))

        # Format dense text chunk for RAG retrieval
        doc_text = f"ICD-10 Code: {code}\nDescription: {desc}"
        if ex1:
            doc_text += f"\nExcludes1 (Must NOT code together): {ex1}"
        if ex2:
            doc_text += f"\nExcludes2 (Not included here): {ex2}"

        documents.append(doc_text)
        metadata.append(item)

    print(f"Generating embeddings for {len(documents)} rules...")
    embeddings = model.encode(
        documents, show_progress_bar=True, normalize_embeddings=True
    )

    # Initialize FAISS Index using Cosine Similarity (Inner Product on normalized embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype(np.float32))

    faiss_path = output_dir / "icd10_rules.index"
    meta_path = output_dir / "icd10_metadata.pkl"

    # Save FAISS Index and Metadata lookup table
    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[✔] Saved FAISS index to '{faiss_path}'.")
    print(f"[✔] Saved metadata mapping to '{meta_path}'.")