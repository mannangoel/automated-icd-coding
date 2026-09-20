# scripts/6_evaluate_benchmark.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ttppr.pipeline import AutomatedCodingPipeline

def run_evaluation_benchmark(
    notes_csv: str = "data/raw/mimic/discharge.csv",
    labels_csv: str = "data/raw/mimic/diagnosis.csv",
):
    print("==================================================")
    print(" Running Grid Search Benchmark (top_k & Thresholds)")
    print("==================================================")
    
    pipeline = AutomatedCodingPipeline()
    
    notes_df = pd.read_csv(notes_csv)
    labels_df = pd.read_csv(labels_csv)
    
    grouped_ground_truth = labels_df.groupby("hadm_id")["icd_code"].apply(set).to_dict()
    all_vocab_codes = sorted(list(pipeline.code2idx.keys()))
    
    print(f"\nEvaluating pipeline on {len(notes_df)} clinical notes...\n")
    
    test_thresholds = [0.20, 0.25, 0.30, 0.45, 0.60, 0.75, 0.85]
    test_top_k = [3, 5]
    
    best_overall_f1 = 0.0
    best_config = {}

    for k in test_top_k:
        print(f"================ Candidate Window: top_k = {k} ================")
        for threshold in test_thresholds:
            y_true_all = []
            y_pred_all = []
            clean_compliance = 0
            total_samples = 0
            
            for _, row in notes_df.iterrows():
                hadm_id = row["hadm_id"]
                text = str(row["text"])
                
                ground_truth = grouped_ground_truth.get(hadm_id, set())
                true_vector = [1 if code in ground_truth else 0 for code in all_vocab_codes]
                
                output = pipeline.process_note(text, top_k=k, threshold=threshold)
                approved_codes = set(output["approved_codes"])
                
                if not output["conflicts_detected"]:
                    clean_compliance += 1
                    
                pred_vector = [1 if code in approved_codes else 0 for code in all_vocab_codes]
                
                y_true_all.append(true_vector)
                y_pred_all.append(pred_vector)
                total_samples += 1

            y_true_arr = np.array(y_true_all)
            y_pred_arr = np.array(y_pred_all)
            
            micro_f1 = f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
            macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            precision = precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
            recall = recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
            ccr = (clean_compliance / total_samples) * 100

            if micro_f1 > best_overall_f1:
                best_overall_f1 = micro_f1
                best_config = {"top_k": k, "threshold": threshold, "precision": precision, "recall": recall, "micro_f1": micro_f1, "macro_f1": macro_f1}

            print(f" top_k={k} | Threshold={threshold:.2f} -> Precision: {precision:.4f} | Recall: {recall:.4f} | Micro-F1: {micro_f1:.4f} | Compliance: {ccr:.1f}%")

    print("\n" + "=" * 60)
    print(" OPTIMAL PRODUCTION CONFIGURATION")
    print("=" * 60)
    print(f" Best top_k Window    : {best_config.get('top_k')}")
    print(f" Best Threshold       : {best_config.get('threshold')}")
    print(f" Micro-Precision      : {best_config.get('precision'):.4f}")
    print(f" Micro-Recall         : {best_config.get('recall'):.4f}")
    print(f" Peak Micro-F1 Score  : {best_config.get('micro_f1'):.4f}")
    print(f" Peak Macro-F1 Score  : {best_config.get('macro_f1'):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation_benchmark()