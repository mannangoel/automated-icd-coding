# scripts/8_train_stage1.py
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ttppr.stage1_dl.dataset import ClinicalICDDataset
from src.ttppr.stage1_dl.model import LAAT

class FocalLoss(nn.Module):
    """Focal Loss for Multi-Label Classification to address severe class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()

def train_model(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    save_path: str = "data/processed/stage1_laat_fine_tuned.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    dataset = ClinicalICDDataset(
        notes_csv="data/raw/mimic/discharge.csv",
        labels_csv="data/raw/mimic/diagnosis.csv",
        code_map_path="data/processed/code_vocab.json",
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LAAT(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        num_classes=dataset.num_classes,
    ).to(device)

    # Safely freeze lower transformer layers if present in parameter names
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            try:
                layer_num = int(name.split(".")[3])
                if layer_num < 8:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            except (IndexError, ValueError):
                pass

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\nStarting fine-tuning across {len(dataset)} records for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete | Loss: {avg_loss:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[✔] Fine-tuned model saved to '{save_path}'.")

if __name__ == "__main__":
    train_model()