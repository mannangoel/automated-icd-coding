# scripts/2_build_index.py
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ttppr.stage2_rag.index_builder import build_icd10_vector_store

if __name__ == "__main__":
    json_input_path = "data/processed/icd10_rules_2027.json"
    output_directory = "data/processed"

    print("Building FAISS Vector Store for Stage 2 RAG Engine...")
    build_icd10_vector_store(json_input_path, output_directory)