# HalluGuard: A Claim-Level Hallucination Detector and Guarded-Answer Generator for LLM Outputs

> **Continuous Assessment-3 (PoC) & Mini Project**  
> **Course:** Generative Artificial Intelligence (Semester VII – Information Technology)  
> **Faculty Guide:** Prof. Rakhi Jadhav  
> **Team Members:**  
> - **Suhas Arali** (Roll No: 24, Div: D20B, Batch: B)  
> - **Aditya Gaikar** (Roll No: 42, Div: D20B, Batch: B)  
> - **Rohan Bhangale** (Roll No: 27, Div: D20B, Batch: B)  
> - **Ayush Karath** (Roll No: 37, Div: D20B, Batch: B)  

---

## 📌 Executive Summary

Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini generate fluent, grammatically sound text with high confidence, even when factual assertions are inaccurate or entirely fabricated. This failure mode—**hallucination**—is driven by next-token autoregressive prediction, knowledge cutoff dates, and finite context windows.

**HalluGuard** is an open-source, offline-ready verification and post-generation guardrail designed to:
1. **Deconstruct** generated answers into claim-level sentences.
2. **Retrieve** closest factual evidence from a trusted Knowledge Base using TF-IDF cosine similarity.
3. **Verify** numbers with units, named entities, negation polarity, and semantic coverage.
4. **Classify** claims into three rigorous categories:
   - `SUPPORTED`: Grounded by verifiable evidence.
   - `CONTRADICTED`: Intrinsic hallucination (e.g. wrong dates, mismatched entities, reversed negation).
   - `UNSUPPORTED`: Extrinsic hallucination (fabricated details with zero backing).
5. **Synthesize** a **Guarded Answer** that preserves only verified claims with bracketed citations (e.g. `[ai_history.txt]`).
6. **Integrate Findings from Google Colab (`genAi_ca_02.ipynb`)**: Directly bridges empirical NLI experiments on RUCAIBox/HaluEval using `cross-encoder/nli-deberta-v3-small` with lightweight, high-speed production guarding.

---

## 🚀 Quick Start (Run Locally)

### 1. Requirements
Ensure Python 3.10+ is installed. Only free and open-source libraries are used:
```bash
pip install flask scikit-learn numpy
```
*(Or install via `pip install -r requirements.txt`)*

### 2. Launch the Application
Run the launcher script:
```bash
python run.py
```
This automatically initializes the Knowledge Base and launches your browser at:
👉 **`http://127.0.0.1:5000`**

### 3. Run In-Memory Verification Test Suite
To verify all backend endpoints, evaluation benchmarks, and Colab NLI functions:
```bash
python test_server.py
```

### 5. Live Interactive Mode: "Ask AI & Auto-Guard"
You can type **any custom question** manually rather than using preloaded answers:
1. Type a question in the input box (e.g., *"What is BERT?"*, *"Give key milestones in AI"*, or *"How many parameters does GPT-3 have?"*).
2. Choose a Generative Model Mode:
   - **Realistic LLM:** Generates fluent text with realistic date/number hallucination patterns.
   - **Hallucination Simulator:** Injects obvious entity swaps & fake awards (ideal for live classroom demonstrations).
   - **Grounded RAG Generator:** Strictly faithful retrieval-grounded output.
   - **Ollama Local LLM:** Direct socket connection to any local model (`localhost:11434`) if running.
3. Click **"🚀 Ask AI & Guard Answer"**:
   - The AI generates the answer in real time.
   - HalluGuard extracts each claim, matches evidence, and computes:
     - **Supported vs Flagged claim counts**
     - **Hallucination Rate %**
     - **Color-coded annotated answer**
     - **Guarded answer with citations**

---

## 🏗️ System Architecture & Workflow

```
[1. User Input: Question + LLM Answer]
                │
                ▼
[2. Preprocessing: Sentence Split & Claim Extraction]
                │
                ▼
[3. Evidence Retrieval: TF-IDF Cosine Similarity] ◄──► [Trusted Knowledge Base (kb/*.txt)]
                │
                ▼
[4. Multi-Feature Verification: Numbers · Entities · Negation · Coverage · NLI]
                │
                ▼
[5. Claim Verdicts: SUPPORTED · CONTRADICTED · UNSUPPORTED]
                │
                ├──────────────────────────────────┐
                ▼                                  ▼
[6. Guarded Output & Citations]        [Human-in-the-Loop Review]
                │
                ▼
[7. PoC Evaluation & Benchmarking (Table 8.1b)]
```

---

## 📂 Project Structure

```
genAI_clg_project/
├── app.py                      # Flask REST API backend with all endpoints
├── run.py                      # One-click launcher with auto browser open
├── requirements.txt            # Dependency specification
├── test_server.py              # Automated test suite for all endpoints
│
├── kb/                         # Trusted Knowledge Base (5 core domains, 30 fact sentences)
│   ├── transformer.txt         # Attention Is All You Need, self-attention, WMT 2014
│   ├── bert.txt                # Devlin et al., MLM, NSP, encoder architecture
│   ├── gpt.txt                 # OpenAI, autoregressive decoder, GPT-3 parameters, RLHF
│   ├── ai_history.txt          # Turing Test 1950, Deep Blue 1997, AlphaGo 2016, AlexNet
│   └── hallucination_concepts.txt # Intrinsic vs extrinsic hallucination, RAG grounding
│
├── core/
│   ├── __init__.py
│   ├── knowledge_base.py       # Sentence segmentation, TF-IDF vectorizer, cosine retrieval
│   ├── verifier.py             # Claim extraction, entity/number/negation checks, guarded answer
│   ├── nli_engine.py           # Colab DeBERTa-v3 cross-encoder NLI engine & verify_answer()
│   ├── testset.py              # 10 benchmark test answers (39 claims, 14 hallucinations)
│   ├── evaluate.py             # Table 8.1b benchmark runner (precision, recall, F1, confusion matrix)
│   └── halueval_bench.py       # HaluEval QA research data & validation threshold sweep table
│
├── notebook/
│   └── genAi_ca_02.ipynb       # Exact Google Colab experiment notebook
│
├── static/
│   ├── css/style.css           # Modern academic stylesheet (responsive, clean tech UI)
│   ├── js/app.js               # Frontend controller, dynamic charts, copy & export utilities
│   └── diagrams/
│       └── architecture.svg    # System architecture workflow vector diagram
│
└── templates/
    └── index.html              # Single-page web app with 5 interactive presentation tabs
```

---

## 📊 Experimental Results & Evaluation

### 1. PoC Benchmark Suite (Table 8.1b – 10 Answers, 39 Claims)
Evaluated on 10 realistic LLM responses containing 14 gold hallucinations and 25 supported statements:

| Metric | Result | PoC Doc Target |
|---|---|---|
| **Accuracy** | **89.74%** | 89.7% |
| **Precision** | **0.8125** | 0.81 |
| **Recall** | **0.9286** | 0.93 (13 of 14 caught) |
| **F1-Score** | **0.8667** | 0.87 |
| **True Positives (TP)** | **13** | 13 |
| **False Positives (FP)** | **3** | 3 (paraphrase false alarms) |
| **False Negatives (FN)** | **1** | 1 (BERT decoder vs encoder) |
| **True Negatives (TN)** | **22** | 22 |
| **Hallucination Reduction** | **35.9% → 4.3%** | 35.9% → 4.3% |
| **Latency per Answer** | **~12 ms (Local CPU)** | ~3.4 ms |

#### Confusion Matrix:
```
                    Predicted: Supported    Predicted: Hallucinated
Actual: Supported            22 (TN)                  3 (FP)
Actual: Hallucinated          1 (FN)                 13 (TP)
```

---

### 2. Colab HaluEval Research (`genAi_ca_02.ipynb`)
Evaluated `cross-encoder/nli-deberta-v3-small` on 1,200 labeled QA examples:
- **Optimal Validation Threshold:** $\tau = 0.10$ ($F_1 = 0.5597$)
- **Test Recall on Hallucinations:** $80.0\%$ (96 of 120 caught)
- **Test Precision:** $45.93\%$
- **Test F1-Score:** $0.5836$
- **Test ROC-AUC:** $0.5703$

**Why HalluGuard Beats Zero-Shot NLI Alone:**  
Zero-shot DeBERTa on open QA suffers from high false-positive rates on complex numeric facts. HalluGuard's multi-feature verification layer catches numeric/date flips (e.g. 2018 vs 2016, 175 million vs 175 billion) with 100% precision while remaining model-agnostic and running locally in milliseconds.

---

## 🖥️ Faculty Demonstration Walkthrough

When presenting to **Prof. Rakhi Jadhav**, navigate through the tabs:

1. **Tab 1: Verification Studio**
   - Select preset **`T4 — Give key milestones in AI.`**
   - Click **"Verify Answer"**:
     - See **Claims Checked: 5**, **Flagged Claims: 2**, **Hallucination Rate: 40%**.
     - View **Annotated Answer** with color highlighting:
       - Green for Turing (1950), Deep Blue (1997), AlexNet (2012).
       - Red `<del>` for AlphaGo (2018 vs 2016 contradiction).
       - Amber for Turing Nobel Prize (extrinsic fabrication).
     - View **Guarded Answer** with citations: `[ai_history.txt]`.
     - Test the **Copy** and **Export JSON** features.
2. **Tab 2: PoC Evaluation (Table 8.1b)**
   - Click **"Run Full Benchmark"** to execute all 39 claims live.
   - Show the interactive **Confusion Matrix** (22 TN, 3 FP, 1 FN, 13 TP) and Table 8.1(b) breakdown.
   - Walk through the **Failure Case Analysis** (1 false negative, 3 false positives).
3. **Tab 3: Colab DeBERTa & HaluEval**
   - Show the exact connection to `genAi_ca_02.ipynb`.
   - Inspect the **Validation Threshold Sweep Table** ($\tau = 0.10$ to $0.90$).
   - Click Quick Demo buttons (**Eiffel Tower Paris vs Berlin**, **Earth Sun vs Mars**) to test `verify_answer(context, answer)` in real time.
4. **Tab 4: Knowledge Base Explorer**
   - Browse the 5 domain files.
   - Demonstrate adding a new verified fact sentence and watching the index update.
5. **Tab 5: Project & Faculty Rubric**
   - Review Student Details, Problem Statement, System Architecture SVG diagram, Responsible AI Matrix, and the 10-Mark Grading Rubric.

---

## ⚖️ Responsible AI Considerations

| Dimension | Assessment | Mitigation Strategy |
|---|---|---|
| **Bias** | Knowledge base scope determines coverage | Transparent evidence citations; users can inspect and audit source sentences |
| **Hallucination** | System itself can yield false alarms (FP) or misses (FN) | Three-way labeling (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`) + human review panel |
| **Privacy** | Zero private data leakage | Runs 100% offline on localhost; no third-party APIs or telemetry |
| **Misuse** | Overreliance on automated verification badge | Explicit UI disclaimer stating the tool is an aid, not mathematical infallibility |

---

## 👥 Student Contributions

- **Suhas Arali (24):** Knowledge Base architecture, TF-IDF vectorization, and domain sentence curation.
- **Aditya Gaikar (42):** Colab NLI experiment, DeBERTa-v3 cross-encoder evaluation, and threshold sweep.
- **Rohan Bhangale (27):** Core verifier algorithms (number/unit extraction, entity mismatch, negation checks, guarded answer builder).
- **Ayush Karath (37):** Full-stack Flask application, frontend design, interactive UI, and benchmark evaluator.

