# src/ttppr/pipeline.py
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import chromadb
from transformers import AutoTokenizer

# 1. Import your true model architecture
from src.ttppr.stage1_dl.model import LAAT

from langchain_groq import ChatGroq

load_dotenv()


class AutomatedCodingPipeline:
    def __init__(
        self,
        vocab_path: str = "data/processed/code_vocab.json",
        weights_path: str = "data/processed/stage1_laat_fine_tuned.pt",
        chroma_path: str = "data/processed/chroma_db",
        collection_name: str = "icd10_guidelines",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_path = Path(vocab_path)
        self.weights_path = Path(weights_path)
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name

        # 1. Load Vocabulary Map
        self._load_vocab()

        # 2. Load Tokenizer & Model Weights
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._load_stage1_model()

        # 3. Connect ChromaDB Client
        self._connect_chromadb()

        # 4. Initialize ChatGroq Model
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    def _load_vocab(self):
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found at {self.vocab_path}")
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        # Backward compatibility for the evaluation benchmark
        self.code2idx = self.vocab 
        
        self.idx_to_code = {v: k for k, v in self.vocab.items()}
        self.num_classes = len(self.vocab)

    def _load_stage1_model(self):
        # Instantiate your actual LAAT model from stage1_dl
        self.model = LAAT(model_name=self.model_name, num_classes=self.num_classes)
        
        if self.weights_path.exists():
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                # strict=True guarantees the trained weights perfectly align with the model layers
                self.model.load_state_dict(state_dict, strict=True)
                print(f"Loaded Stage 1 fine-tuned weights from {self.weights_path}")
            except Exception as e:
                print(f"[!] Warning loading weights: {e}. Running with base initialization.")
        else:
            print(f"[!] Warning: Weights checkpoint {self.weights_path} not found.")
            
        self.model.to(self.device)
        self.model.eval()

    def _connect_chromadb(self):
        if not self.chroma_path.exists():
            print(f"[!] Warning: ChromaDB path {self.chroma_path} does not exist.")
            self.collection = None
            return
        
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        try:
            self.collection = client.get_collection(name=self.collection_name)
            print(f"Connected to ChromaDB collection: '{self.collection_name}'")
        except Exception:
            self.collection = None
            print(f"[!] Collection '{self.collection_name}' not found in ChromaDB.")

    # --- Stage 1 Inference ---
    def _predict_stage1(self, text: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(logits).squeeze(0)

        top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))

        candidates = []
        for prob, idx in zip(top_probs, top_indices):
            p_val = prob.item()
            if p_val >= threshold:
                code = self.idx_to_code[idx.item()]
                candidates.append({
                    "code": code,
                    "confidence": f"{p_val:.4f}"
                })
        return candidates

    # --- Stage 2 RAG Retrieval ---
    def _retrieve_rules(self, candidate_codes: List[str]) -> List[Dict[str, Any]]:
        retrieved_rules = []
        if not self.collection:
            # Fallback mock rule metadata if database is unavailable
            for code in candidate_codes:
                retrieved_rules.append({
                    "code": code,
                    "description": f"Standard ICD-10 clinical diagnosis for code {code}",
                    "excludes1": None
                })
            return retrieved_rules

        # Create both with-dot and without-dot variations to ensure a match
        clean_ids = [c.replace(".", "").strip() for c in candidate_codes]
        all_query_ids = list(set(candidate_codes + clean_ids))
        
        results = self.collection.get(ids=all_query_ids)

        found_map = {}
        if results and results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                icd_code = meta.get("icd_code")
                found_map[icd_code] = {
                    "code": icd_code,
                    "description": meta.get("description", doc),
                    "excludes1": meta.get("excludes1", None)
                }
                # Also store a dotless mapping just in case
                found_map[icd_code.replace(".", "")] = found_map[icd_code]

        # Format retrieved items matching candidate order
        for original_code, clean_id in zip(candidate_codes, clean_ids):
            if original_code in found_map:
                retrieved_rules.append(found_map[original_code])
            elif clean_id in found_map:
                retrieved_rules.append(found_map[clean_id])
            else:
                retrieved_rules.append({
                    "code": original_code,
                    "description": "No explicit CMS guidelines found in database.",
                    "excludes1": None
                })

        return retrieved_rules

    # --- Stage 3 Symbolic Verification ---
    def _verify_symbolic_and_llm(
        self, clinical_note: str, candidates: List[Dict[str, Any]], rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        # 1. Symbolic Excludes1 Conflict Checking
        conflicts = []
        approved_codes = []
        rule_dict = {r["code"]: r for r in rules}

        for cand in candidates:
            code = cand["code"]
            rule_info = rule_dict.get(code, {})
            excludes = rule_info.get("excludes1")
            
            # If Excludes1 mutually exclusive codes exist in candidate list
            if excludes:
                conflict_found = False
                for other in candidates:
                    if other["code"] != code and other["code"] in excludes:
                        conflicts.append(f"{code} mutually excludes {other['code']}")
                        conflict_found = True
                if not conflict_found:
                    approved_codes.append(code)
            else:
                approved_codes.append(code)

        return {
            "approved_codes": approved_codes,
            "conflicts_detected": conflicts if conflicts else None
        }

    # --- Process Note Pipeline Handler ---
    def process_note(self, clinical_note: str, top_k: int = 3, threshold: float = 0.0) -> Dict[str, Any]:
        # Stage 1: Neural Candidate Extraction
        stage1_candidates = self._predict_stage1(clinical_note, top_k=top_k, threshold=threshold)
        candidate_codes = [c["code"] for c in stage1_candidates]

        # Stage 2: RAG Guidelines Retrieval
        retrieved_rules = self._retrieve_rules(candidate_codes)

        # Stage 3: Verification Logic
        verification = self._verify_symbolic_and_llm(clinical_note, stage1_candidates, retrieved_rules)

        return {
            "stage1_candidates": stage1_candidates,
            "retrieved_rules": retrieved_rules,
            "approved_codes": verification["approved_codes"],
            "conflicts_detected": verification["conflicts_detected"],
        }