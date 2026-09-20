# scripts/download_hf_dataset.py
import csv
from pathlib import Path
from datasets import load_dataset

def download_open_icd_data(output_dir: str = "data/raw/mimic"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    notes_file = out_path / "discharge.csv"
    diagnosis_file = out_path / "diagnosis.csv"
    
    print("Fetching the COMPLETE MedSynth ICD-10 dataset from Hugging Face...")
    dataset = load_dataset("Ahmad0067/MedSynth", split="train")
    
    discharge_rows = []
    diagnosis_rows = []
    
    for i, row in enumerate(dataset):
        hadm_id = 4000000 + i
        
        # The dataset contains 4 columns: Note, Dialogue, ICD10, ICD10_desc
        # We will extract values directly to avoid any dictionary key formatting errors
        values = list(row.values())
        
        if len(values) >= 3:
            text = str(values[0]).strip()      # 1st column: Note
            icd_code = str(values[2]).strip()  # 3rd column: ICD10
            
            if text and icd_code and text.lower() != "nan" and icd_code.lower() != "nan":
                discharge_rows.append({"hadm_id": hadm_id, "text": text})
                diagnosis_rows.append({"hadm_id": hadm_id, "icd_code": icd_code})

    with open(notes_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hadm_id", "text"])
        writer.writeheader()
        writer.writerows(discharge_rows)

    with open(diagnosis_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hadm_id", "icd_code"])
        writer.writeheader()
        writer.writerows(diagnosis_rows)

    print(f"[✔] Successfully downloaded and processed {len(discharge_rows)} clinical notes.")

if __name__ == "__main__":
    download_open_icd_data()