# src/ttppr/stage2_rag/lcd_retriever.py
import pandas as pd
from pathlib import Path

class LCDPolicyRetriever:
    """Retrieves Local Coverage Determination (LCD) policies for candidate ICD-10 codes."""

    def __init__(self, lcd_dir: str | Path = "data/raw/lcd"):
        self.lcd_dir = Path(lcd_dir)
        self.lcd_df = None
        self.mapping_df = None
        self.is_loaded = False
        self._load_data()

    def _load_data(self):
        lcd_path = self.lcd_dir / "lcd.csv"
        
        # Check potential filenames for the LCD-to-ICD-10 crosswalk file
        mapping_candidates = [
            self.lcd_dir / "lcd_x_icd10_code.csv",
            self.lcd_dir / "lcd_x_icd10.csv",
            self.lcd_dir / "lcd_icd10_rel.csv"
        ]
        
        mapping_path = next((p for p in mapping_candidates if p.exists()), None)

        if not lcd_path.exists() or not mapping_path:
            print(f"[Warning] LCD CSV files not found in {self.lcd_dir}. Proceeding with stubbed policy lookup.")
            return

        try:
            self.lcd_df = pd.read_csv(lcd_path, low_memory=False)
            self.mapping_df = pd.read_csv(mapping_path, low_memory=False)
            
            # Normalize column names to lowercase
            self.lcd_df.columns = [c.lower() for c in self.lcd_df.columns]
            self.mapping_df.columns = [c.lower() for c in self.mapping_df.columns]
            self.is_loaded = True
            print(f"[✔] Successfully loaded {len(self.lcd_df)} LCD policies and {len(self.mapping_df)} ICD-10 crosswalk records.")
        except Exception as e:
            print(f"[Warning] Error loading LCD CSVs: {e}. Falling back to default policy checks.")

    def get_policies_for_code(self, icd10_code: str) -> list[dict]:
        """Returns associated LCD coverage policy IDs and titles for a given ICD-10 code."""
        if not self.is_loaded:
            # Fallback mock response for testing when raw files are unpopulated
            return [{
                "lcd_id": "L33580",
                "title": "Coverage Policy for End-Stage Renal Disease & Chronic Kidney Monitoring",
                "status": "Active"
            }]

        clean_code = icd10_code.replace(".", "").strip().upper()
        
        # Find matching LCD IDs in crosswalk table
        code_col = next((c for c in self.mapping_df.columns if "icd" in c or "code" in c), None)
        lcd_id_col = next((c for c in self.mapping_df.columns if "lcd" in c), None)

        if not code_col or not lcd_id_col:
            return []

        matched_mapping = self.mapping_df[
            self.mapping_df[code_col].astype(str).str.replace(".", "").str.strip().str.upper() == clean_code
        ]

        if matched_mapping.empty:
            return []

        matched_lcd_ids = matched_mapping[lcd_id_col].unique()

        # Retrieve policy details from master LCD table
        lcd_id_master_col = next((c for c in self.lcd_df.columns if "lcd_id" in c or "id" in c), None)
        title_col = next((c for c in self.lcd_df.columns if "title" in c or "name" in c), None)

        results = []
        for lcd_id in matched_lcd_ids:
            policy_row = self.lcd_df[self.lcd_df[lcd_id_master_col] == lcd_id]
            title = policy_row[title_col].values[0] if not policy_row.empty and title_col else "Policy Details Available"
            results.append({
                "lcd_id": str(lcd_id),
                "title": str(title),
                "status": "Active"
            })

        return results