# Automated Clinical ICD-10 Coding & Verification Pipeline

This repository implements an enterprise-grade, multi-stage hybrid ML/DL and RAG architecture for automated ICD-10 medical coding.

Medical coding is highly susceptible to human error and complex regulatory rules, often resulting in costly claim denials and revenue leakage. Conversely, pure generative AI models are prone to hallucinating medical codes and lack deterministic rule awareness. This system bridges that gap by combining probabilistic deep learning with strict Retrieval-Augmented Generation (RAG) and generative LLM auditing, providing a reliable, compliance-first engine for clinical documentation and medical billing.

## Stakeholder Value & Business Impact

* **Reduction in Claim Denials:** Automatically flags `Excludes1` (mutually exclusive) billing conflicts before they are submitted.

* **Auditable AI:** Eliminates the "black box" AI problem by forcing the LLM to extract and cite a direct 1-sentence quote from the patient's chart for every approved code.

* **Regulatory Grounding:** Decisions are dynamically cross-referenced against 47,025+ official Centers for Medicare & Medicaid Services (CMS) guidelines rather than relying on model memorization.

## System Architecture

The system operates in three distinct, sequential stages to ensure absolute clinical accuracy and adherence to government billing regulations:

1. **Stage 1: Neural Candidate Extraction (Probabilistic)**

   * **Model:** `emilyalsentzer/Bio_ClinicalBERT` fine-tuned with a Label-Attention (LAAT) classification head.

   * **Function:** Ingests unstructured clinical notes and projects probability scores across 2,037 unique ICD-10 diagnostic classes, identifying the most likely conditions.

2. **Stage 2: RAG Guidelines Retrieval (Deterministic)**

   * **Knowledge Base:** A persistent ChromaDB vector store indexing official CMS guidelines and medical descriptions.

   * **Function:** Uses the Stage 1 candidate predictions as metadata filters to instantly retrieve the exact legal and clinical definitions required for auditing.

3. **Stage 3: Symbolic & Generative Auditing (LLM Guardrail)**

   * **Engine:** `openai/gpt-oss-20b` via the Groq API + Symbolic constraint logic.

   * **Function:** Cross-references the clinical note, the predicted codes, and the official CMS rules. It forcibly rejects unsupported codes and generates a structured compliance report detailing approved codes and their clinical justifications.

## Project Structure

```
├── data/
│   ├── raw/                 # Raw clinical CSVs (Ignored in Git)
│   └── processed/           # chroma_db/, code_vocab.json, trained .pt weights
├── scripts/
│   ├── 5_orchestrate_end_to_end.py  # Main pipeline execution script
│   ├── 6_evaluate_benchmark.py      # Grid-search threshold evaluation
│   └── 8_train_stage1.py            # Local model training script
├── src/ttppr/
│   ├── pipeline.py          # Core AutomatedCodingPipeline class
│   └── stage1_dl/
│       ├── dataset.py       # PyTorch Dataset & Tokenization logic
│       └── model.py         # BioClinical-BERT + LAAT Architecture
├── .env                     # API keys (Groq, HF)
├── pyproject.toml           # uv project configuration
└── README.md

```

##  Installation & Setup

This project utilizes `uv` for lightning-fast dependency management and environment resolution.

**Clone the repository:**

```
git clone https://github.com/mannangoel/automated-icd-coding.git
cd automated-icd-coding

```

**Sync the environment:**

```
uv sync

```

**Environment Variables:**
Create a `.env` file in the root directory and add your API keys:

```
GROQ_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional for higher rate limits

```

## Usage

### 1. Run the End-to-End Pipeline

To test the full 3-stage architecture on a sample clinical note and view the final LLM audit report:

```
uv run python -m scripts.5_orchestrate_end_to_end

```

### 2. Run the Benchmark Evaluator

To evaluate the pipeline's performance across a test set using thresholding and grid search:

```
uv run python -m scripts.6_evaluate_benchmark

```

## Acknowledgements

* **Base Model:** Bio_ClinicalBERT by Emily Alsentzer et al.

* **Data Sources:** MIMIC / MedSynth subsets for domain adaptation.
