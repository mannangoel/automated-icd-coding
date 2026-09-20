# scripts/3_predict_stage1.py
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ttppr.stage1_dl.dataset import ClinicalICDDataset
from src.ttppr.stage1_dl.model import LAAT

if __name__ == "__main__":
    notes_csv = "data/raw/mimic/discharge.csv"
    labels_csv = "data/raw/mimic/diagnoses_icd.csv"
    code_map_path = "data/processed/code_vocab.json"

    print("Loading Dataset and Tokenizing Clinical Notes...")
    dataset = ClinicalICDDataset(notes_csv, labels_csv, code_map_path)
    print(f"[✔] Dataset initialized. Total Classes/ICD-Codes: {dataset.num_classes}")

    print("Initializing BioClinical-BERT + LAAT Model Architecture...")
    model = LAAT(model_name="emilyalsentzer/Bio_ClinicalBERT", num_classes=dataset.num_classes)
    model.eval()

    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0)
    attention_mask = sample["attention_mask"].unsqueeze(0)

    print("Executing forward pass candidate prediction...")
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits).squeeze(0)

    # Top-K Candidate Extraction
    k = min(3, dataset.num_classes)
    top_k_probs, top_k_indices = torch.topk(probabilities, k=k)

    idx2code = {v: k for k, v in dataset.code2idx.items()}
    print(f"\n[✔] Top-{k} Candidate Predictions for Admission ID {sample['hadm_id']}:")
    for prob, idx in zip(top_k_probs, top_k_indices):
        code = idx2code[idx.item()]
        print(f"  - Code: {code:<10} Confidence Probability: {prob.item():.4f}")