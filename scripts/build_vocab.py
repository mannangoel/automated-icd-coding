# scripts/build_vocab.py
import csv
import json
from pathlib import Path

def build_vocabulary(
    diagnosis_csv: str = "data/raw/mimic/diagnosis.csv",
    output_json: str = "data/processed/code_vocab.json"
):
    print(f"Scanning {diagnosis_csv} to build code vocabulary...")
    
    unique_codes = set()
    with open(diagnosis_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["icd_code"]:
                unique_codes.add(row["icd_code"])
                
    sorted_codes = sorted(list(unique_codes))
    code2idx = {code: idx for idx, code in enumerate(sorted_codes)}
    
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(code2idx, f, indent=4)
        
    print(f"[✔] Vocabulary built with {len(code2idx)} unique ICD-10 codes.")
    print(f"[✔] Saved to {output_json}")

if __name__ == "__main__":
    build_vocabulary()