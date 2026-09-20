# scripts/build_knowledge_base.py
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

def build_chroma_knowledge_base(
    json_rules_path: str = "data/processed/icd10_rules_2027.json",
    db_path: str = "data/processed/chroma_db",
    collection_name: str = "icd10_guidelines",
    batch_size: int = 1000
):
    json_file = Path(json_rules_path)
    if not json_file.exists():
        print(f"[X] Could not find {json_rules_path}. Make sure 1_parse_xml.py has been run.")
        return

    print(f"Loading parsed ICD-10 rules from {json_rules_path}...")
    with open(json_file, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    # Extract documents, metadata, and IDs
    docs = []
    metadatas = []
    ids = []

    # Handle either dict format {code: desc} or list format [{"code": ..., "desc": ...}]
    if isinstance(rules_data, dict):
        items = rules_data.items()
    elif isinstance(rules_data, list):
        items = [(item.get("code"), item.get("description", item.get("desc", ""))) for item in rules_data]
    else:
        print("[X] Unrecognized format in JSON file.")
        return

    for code, desc in items:
        if code and desc:
            clean_code = str(code).strip()
            clean_desc = str(desc).strip()
            
            docs.append(f"ICD-10 Code {clean_code}: {clean_desc}")
            metadatas.append({"icd_code": clean_code, "description": clean_desc})
            ids.append(clean_code)

    print(f"Loaded {len(docs)} ICD-10 code records into memory.")

    # Initialize ChromaDB persistent client
    print(f"Initializing persistent ChromaDB at '{db_path}'...")
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Re-create collection for a clean build
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Load local SentenceTransformer embedder
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating vector embeddings and inserting into ChromaDB...")
    total_samples = len(docs)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        
        batch_docs = docs[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]

        # Generate embeddings
        embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()

        # Insert batch into ChromaDB
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"  -> Processed [{end_idx}/{total_samples}] records...")

    print(f"\n[✔] ChromaDB successfully created at '{db_path}' with {len(docs)} indexed codes.")

if __name__ == "__main__":
    build_chroma_knowledge_base()