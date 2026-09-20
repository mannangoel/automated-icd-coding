import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ttppr.stage2_rag.retriever import ICD10Retriever
from src.ttppr.stage3_llm.verifier import ClinicalVerifier

if __name__ == "__main__":
    index_path = "data/processed/icd10_rules.index"
    meta_path = "data/processed/icd10_metadata.pkl"

    
    stage1_candidates = ["E11.9", "N18.3", "I10"]

    print("--- STAGE 2: RAG Context Retrieval ---")
    retriever = ICD10Retriever(index_path, meta_path)
    retrieved_rules = retriever.get_rules_for_candidates(stage1_candidates)

    for rule in retrieved_rules:
        print(f"Code: {rule['code']} | Desc: {rule['description']}")
        if rule.get("excludes1"):
            print(f"  └─ Excludes1: {rule['excludes1']}")

    print("\n--- STAGE 3: Symbolic Rule Verification ---")
    verification_report = ClinicalVerifier.validate_code_conflicts(
        retrieved_rules
    )

    print(f"Approved Final Codes: {verification_report['approved_codes']}")
    if verification_report["conflicts_detected"]:
        print(f"Conflicts Detected: {verification_report['conflicts_detected']}")
    else:
        print("[✔] No coding rule violations found.")