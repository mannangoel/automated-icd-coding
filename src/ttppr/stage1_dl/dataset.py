import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer

class ClinicalICDDataset(Dataset):
    """
    PyTorch Dataset for Clinical Note to ICD-10 Multi-Label Classification.
    """
    def __init__(self, notes_csv: str | Path, labels_csv: str | Path, code_map_path: str | Path, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", max_length: int = 512):
        self.notes_df = pd.read_csv(notes_csv)
        self.labels_df = pd.read_csv(labels_csv)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Load or generate master ICD code-to-index vocabulary
        if Path(code_map_path).exists():
            with open(code_map_path, 'r') as f:
                self.code2idx = json.load(f)
        else:
            unique_codes = sorted(self.labels_df['icd_code'].unique())
            self.code2idx = {code: idx for idx, code in enumerate(unique_codes)}
            Path(code_map_path).parent.mkdir(parents=True, exist_ok=True)
            with open(code_map_path, 'w') as f:
                json.dump(self.code2idx, f, indent=2)

        self.num_classes = len(self.code2idx)
        
        # Merge notes with aggregated multi-hot labels per admission (hadm_id)
        self.samples = []
        grouped_labels = self.labels_df.groupby('hadm_id')['icd_code'].apply(list).to_dict()
        
        for _, row in self.notes_df.iterrows():
            hadm_id = row['hadm_id']
            text = str(row['text'])
            codes = grouped_labels.get(hadm_id, [])
            
            # Create multi-hot target binary vector
            target_vector = torch.zeros(self.num_classes, dtype=torch.float32)
            for code in codes:
                if code in self.code2idx:
                    target_vector[self.code2idx[code]] = 1.0
                    
            self.samples.append({"text": text, "target": target_vector, "hadm_id": hadm_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "targets": item["target"],
            "hadm_id": item["hadm_id"]
        }