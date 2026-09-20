import os
import sys
import json
import torch
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Add project root to path so we can import the pipeline
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ttppr.pipeline import AutomatedCodingPipeline

load_dotenv()

# --- 1. Load Stage 1 Artifacts & Vocabulary ---
def load_stage1_assets(vocab_path="data/processed/code_vocab.json"):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx_to_code = {v: k for k, v in vocab.items()}
    return vocab, idx_to_code

# --- 2. Retrieve Guidelines from ChromaDB ---
def retrieve_code_guidelines(candidate_codes, db_path="data/processed/chroma_db", collection_name="icd10_guidelines"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    
    # Query ChromaDB using IDs matching Stage 1 candidate codes
    results = collection.get(ids=candidate_codes)
    
    guidelines = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"], results["metadatas"]):
            guidelines.append(f"- Code {meta.get('icd_code')}: {meta.get('description')}")
            
    return "\n".join(guidelines) if guidelines else "No explicit guidelines retrieved."

# --- 3. Stage 2 Generative Verification via ChatGroq ---
def verify_with_groq(clinical_note, candidate_codes, guidelines):
    llm = ChatGroq(
        # Updated to use the 2026 GPT-OSS 20B model as requested
        model="openai/gpt-oss-20b",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    
    prompt = ChatPromptTemplate.from_template("""You are an expert medical coding auditor specializing in ICD-10 classification. Your job is to evaluate candidate ICD-10 codes predicted by a neural classifier against the clinical note text and official CMS guidelines. Provide a clean, structured audit report.

### Patient Clinical Note:
{clinical_note}

### Stage 1 Classifier Candidates:
{candidate_codes}

### Official CMS Guidelines:
{guidelines}

### Task:
1. Validate which candidate codes are explicitly supported by the clinical note.
2. Reject any candidate codes not supported by evidence in the text.
3. For each validated code, provide a brief 1-sentence evidence quote from the note.

Format your output clearly with 'Verified Codes' and 'Justification' sections.
""")
    
    chain = prompt | llm
    response = chain.invoke({
        "clinical_note": clinical_note,
        "candidate_codes": ", ".join(candidate_codes),
        "guidelines": guidelines
    })
    
    return response.content

# --- Main Orchestration Routine ---
def run_full_pipeline(sample_note: str, top_k: int = 5):
    print("\n--- [Stage 1] Neural Fine-Tuned Candidate Prediction ---")
    
    
    pipeline = AutomatedCodingPipeline()
    
   
    stage1_results = pipeline._predict_stage1(sample_note, top_k=top_k)
    candidate_codes = [cand["code"] for cand in stage1_results]
    
    print(f"Top Candidate ICD-10 Codes: {candidate_codes}")
    for cand in stage1_results:
        print(f"  • {cand['code']} (Confidence: {cand['confidence']})")

    print("\n--- [Stage 2] ChromaDB Vector Search & Guideline Retrieval ---")
    retrieved_rules = retrieve_code_guidelines(candidate_codes)
    print("Retrieved Official Guidelines:")
    print(retrieved_rules)

    print("\n--- [Stage 2] Generative Verification via ChatGroq ---")
    final_report = verify_with_groq(sample_note, candidate_codes, retrieved_rules)
    print("\n" + "="*50)
    print(final_report)
    print("="*50)

if __name__ == "__main__":
    sample_text = (
        "Patient presents with acute chest pain and shortness of breath. "
        "Electrocardiogram indicates ST-elevation myocardial infarction (STEMI)."
    )
    run_full_pipeline(sample_text)