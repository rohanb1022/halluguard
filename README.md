# 🛡️ HalluGuard: A Claim-Level Hallucination Detector and Guarded-Answer Generator for LLM Outputs

> **Continuous Assessment-3 (PoC) & Mini Project**  
> **Course:** Generative Artificial Intelligence (Semester VII – Information Technology)  
> **Faculty Guide:** Prof. Rakhi Jadhav  
> **Live Deployment:** [https://halluguard-gamma.vercel.app](https://halluguard-gamma.vercel.app)  
> **Team Members:**  
> - **Suhas Arali** (Roll No: 24, Div: D20B, Batch: B) — *Knowledge Base & TF-IDF Retrieval Architecture*  
> - **Aditya Gaikar** (Roll No: 42, Div: D20B, Batch: B) — *DeBERTa-v3 NLI Cross-Encoder & Colab HaluEval Experiments*  
> - **Rohan Bhangale** (Roll No: 27, Div: D20B, Batch: B) — *Core Lexical Verification Algorithms & Guarded Answer Engine*  
> - **Ayush Karath** (Roll No: 37, Div: D20B, Batch: B) — *Full-Stack Flask App, Vercel Serverless & Benchmark Suite*  

---

## 📑 Table of Contents
1. [Project Overview & Motivation](#-project-overview--motivation)
2. [Genesis: From Colab Experiment (CA-02) to Full-Stack PoC (CA-03)](#-genesis-from-colab-experiment-ca-02-to-full-stack-poc-ca-03)
3. [Technology Stack & Architectural Rationale](#-technology-stack--architectural-rationale)
4. [System Architecture & Workflow (Figure 1)](#-system-architecture--workflow-figure-1)
5. [In-Depth Feature & UI Walkthrough (Every Control & Tab)](#-in-depth-feature--ui-walkthrough-every-control--tab)
6. [Core Algorithms & Mathematical Formulations](#-core-algorithms--mathematical-formulations)
7. [PoC Evaluation & Benchmark Results (Table 8.1b)](#-poc-evaluation--benchmark-results-table-81b)
8. [Comprehensive Viva Q&A Guide (For Faculty Examination)](#-comprehensive-viva-qa-guide-for-faculty-examination)
9. [Responsible AI Governance Matrix](#-responsible-ai-governance-matrix)
10. [Local Setup & Deployment Instructions](#-local-setup--deployment-instructions)

---

## 📌 Project Overview & Motivation

Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini generate fluent, authoritative, and linguistically persuasive responses. However, they frequently suffer from **hallucination**—the generation of statements that are factually false, ungrounded, or contradictory to source reality. Hallucinations arise because:
1. **Next-Token Prediction Objective:** LLMs are trained to maximize probabilistic token likelihood, not factual veracity.
2. **Knowledge Cutoff Boundaries:** Information after training dates is absent from model weights.
3. **Finite Context Windows & Attention Limits:** Models lose fidelity over lengthy passages or subtle multi-hop entity references.

### The Solution: HalluGuard
HalluGuard functions as an automated, post-generation guardrail. Rather than evaluating answers as amorphous paragraphs, HalluGuard operates at the **claim level**:
- It segments text into discrete propositional assertions.
- Retrieves verified evidence from a trusted Knowledge Base.
- Applies multi-feature checks (numbers with magnitude scales, named entities, negation polarity, and semantic coverage).
- Classifies each claim into three grounded labels: `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`.
- Synthesizes a **Guarded Answer** that purifies the LLM output by retaining only verified facts with source citations (e.g. `[ai_history.txt]`).

---

## 🧪 Genesis: From Colab Experiment (CA-02) to Full-Stack PoC (CA-03)

Our project represents a rigorous two-stage research progression:

```
[CA-02: Colab Exploration]                      [CA-03: Full-Stack HalluGuard PoC]
- Dataset: RUCAIBox/HaluEval (1,200 QA pairs)   - Multi-Tiered Verification Layer
- Model: cross-encoder/nli-deberta-v3-small     - 5-Domain Trusted Knowledge Base
- Entailment probability scoring                - TF-IDF Cosine Semantic Retrieval
- Swept threshold: tau = 0.10 (F1: 0.5597)      - Number/Unit, Entity, Negation Checking
- Limitations: GPU heavy, high false alarms     - 89.7% Accuracy, 0.81 Precision, 0.93 Recall
                                                - Ultra-fast local latency (~12 ms)
```

### Why Deep Learning NLI Alone is Insufficient (Findings from Colab)
In `notebook/genAi_ca_02.ipynb`, we evaluated DeBERTa-v3 on HaluEval. While it achieved an **80.0% recall** at threshold $\tau = 0.10$, its precision was only **45.9%**, resulting in an overall accuracy of **42.9%**.
- **The Core Flaw:** Cross-encoders often miss exact numeric discrepancies (e.g., claiming GPT-3 has 175 *million* instead of *billion* parameters) or entity swaps when surrounded by high semantic similarity.
- **The Innovation:** HalluGuard augments deep contextual modeling with fine-grained lexical and symbolic verification, achieving **89.7% accuracy** and reducing hallucinations from **35.9% down to 4.3%**.

---

## 💻 Technology Stack & Architectural Rationale

Every technology was selected according to three strict engineering principles: **100% Free & Open-Source**, **Model-Agnostic Modularity**, and **Serverless Cloud Compatibility**.

| Component | Technology | Rationale & Purpose |
|---|---|---|
| **Programming Language** | Python 3.11 | Standard for AI/NLP with native string manipulation, regex, and typing. |
| **Web Framework** | Flask 3.1 | Lightweight, unopinionated WSGI microframework. Eliminates monolithic bloat and starts in milliseconds on serverless Lambdas. |
| **Information Retrieval** | Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`) | Sublinear TF scaling (`sublinear_tf=True`) and word n-grams (1, 2) capture phrases like *"Alan Turing"* or *"Deep Blue"* with zero GPU cost. |
| **Statistical Metrics** | Scikit-Learn Metrics | Live computation of Confusion Matrix, Precision, Recall, F1, and Accuracy. |
| **Vector Math** | NumPy 1.26 | Efficient array operations for similarity matrix slicing and score sorting. |
| **Syntactic Pattern Matching** | Python `re` & Dataclasses | Custom regex parsing for numbers with scale units (`million`, `billion`), capitalized entities, and negation tokens. |
| **Frontend Architecture** | HTML5, CSS3, Vanilla JavaScript (ES6) | Zero-dependency responsive UI with CSS variables, asynchronous Fetch API calls, dynamic badge injection, and no heavy node build steps. |
| **Cloud Deployment** | Vercel Serverless (`@vercel/python`, `api/index.py`, `vercel.json`) | Global CDN hosting routing requests to auto-scaling AWS Lambda microservices at zero cost. |

---

## 🏛️ System Architecture & Workflow (Figure 1)

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. User / Input: Manual Question OR Pre-Configured Benchmark Preset   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Preprocessing: Sentence Segmentation & Claim Extraction             │
│    - Regex punctuation splitting with abbreviation shielding (et al.)  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Evidence Retrieval (TF-IDF Cosine Similarity)                      │
│    ◄──► Queried against Trusted Knowledge Base (kb/*.txt - 30 facts)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Multi-Feature Verification Engine                                   │
│    - Number & Scale Validation (e.g., 2018 vs 2016; 175M vs 175B)      │
│    - Named Entity Mismatch Checking (e.g., Turing vs Nobel Prize)      │
│    - Negation Polarity Conflict Detection                              │
│    - Content-Word Coverage & Jaccard Overlap Computation               │
│    - DeBERTa-v3 Semantic Cross-Encoder Entailment Calibration         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Grounded Claim Classification                                       │
│    ● SUPPORTED (Verified)  ▲ CONTRADICTED (Intrinsic)  ◆ UNSUPPORTED   │
└─────────────────┬────────────────────────────────────┬─────────────────┘
                  │                                    │
                  ▼                                    ▼
┌────────────────────────────────────┐  ┌────────────────────────────────┐
│ 6. Guarded Output Synthesis        │  │ 7. Human-in-the-Loop Oversight │
│    - Interactive Annotated Badges  │  │    - Responsible AI Governance │
│    - Guarded Answer with Citations │  │    - False Alarm Override      │
│    - Hallucination Rate & Latency  │  │    - KB Fact Contribution      │
└─────────────────┬──────────────────┘  └────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 8. Benchmark Evaluation: Precision (81%), Recall (93%), Accuracy (89%) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ In-Depth Feature & UI Walkthrough (Every Control & Tab)

### 1. Header & Navigation Bar
- **Brand Title & Badge:** Displays `HalluGuard PoC CA-3` (Continuous Assessment-3 Proof of Concept).
- **Status Pills:**
  - `● Live / Deployed Status`: Verifies backend server health.
  - `📚 KB Facts: 30`: Displays total number of verified factual statements indexed across the 5 domains in `kb/`.
  - `⚡ Model: Rule + DeBERTa NLI`: Dual-tier architecture combining rule-based lexical precision with neural contextual reasoning.
- **5 Navigation Tabs:** Seamless single-page tab navigation between Studio, PoC Evaluation, Colab Hub, KB Explorer, and Faculty Rubric.

---

### 2. Tab 1: Verification Studio

#### A. Left Panel (Input Controls)
1. **Quick Question Ideas (Pills):**
   - *Buttons:* `🏆 AI Milestones`, `🧠 What is BERT?`, `🤖 GPT-3 & ChatGPT`, `⚡ Transformers`, `⚠️ AI Hallucinations`.
   - *Why do we have them?* One-click shortcuts for viva demonstrations, enabling instantaneous generation and verification without typing.
2. **Type Your Question Here (Input Box):**
   - Free-form text input allowing the student or faculty member to ask *any* custom prompt.
3. **Generative Model Mode (Dropdown):**
   - *What does this dropdown do?* It controls the generation personality and hallucination tendency of the AI before HalluGuard audits it:
     - `🤖 Realistic LLM`: Generates fluent responses resembling ChatGPT/GPT-3.5 that mix accurate information with common subtle hallucinations (e.g. wrong dates like 2018 for AlphaGo).
     - `⚠️ Hallucination Simulator (Demo Mode)`: Injects bold entity swaps, fake awards, and fictitious dates (e.g., Alan Turing winning the Nobel Prize in 1952) so faculty can immediately witness the detector catching extreme errors.
     - `🎯 Grounded RAG Generator`: Emulates an idealized retrieval-grounded assistant that attempts to remain strictly faithful to source documents.
     - `🦙 Ollama Local Model`: Connects to `localhost:11434` for running local open-weights models (Llama 3, Mistral, Phi-3).
4. **`🚀 Ask AI & Guard Answer` (Primary Action Button):**
   - Executes the complete end-to-end pipeline in one click: generates the AI answer, extracts claims, retrieves evidence, calculates the hallucination score, and renders the guarded answer with citations.
5. **Benchmark Sample Preset (Table 8.1b) Dropdown:**
   - *Why do we use it?* Pre-loads the exact 10 hand-labelled benchmark answers (T1 through T10) from Table 8.1b of your Continuous Assessment-3 PoC document.
6. **LLM Answer Text to Verify (Textarea):**
   - Holds the generated answer text or allows pasting arbitrary text from external chatbots (ChatGPT, Claude, Gemini).
7. **`✨ Generate Answer Only` (Action Link):**
   - Generates the text into the box *without* verifying yet, giving the user an opportunity to inspect, edit, or tamper with the text before running the audit.
8. **Detection Settings & Custom Knowledge Context (Collapsible Details):**
   - *Verification Engine Dropdown:*
     - `Fast Lexical & Rule-Based Verifier (~3.4ms)`: Evaluates numbers, entities, negation, and coverage.
     - `DeBERTa-v3 NLI Cross-Encoder`: Contextual entailment probability modeling from Colab.
     - `Hybrid Ensemble`: Combines lexical precision for numbers with NLI semantic understanding.
   - *Hallucination Threshold Slider ($\tau = 0.10$):*
     - Tuned in Colab on the validation set to maximize F1-score.
   - *Custom Reference Context Checkbox & Textarea:*
     - Allows the user to paste any custom document or ground truth to verify the answer against, instead of the 5 default KB files!
9. **`🔍 Verify Answer Box` Button:**
   - Audits whatever text is currently typed into the textarea.
10. **`Clear All` Button:**
    - Resets all inputs and metric counters.

#### B. Right Panel (Output & Verification Report)
1. **The 4 Metric Cards:**
   - `Claims Checked`: Total count of individual declarative sentences extracted from the answer.
   - `Supported [Verified]`: Count of claims confirmed by evidence.
   - `Flagged Hallucinations`: Count of contradicted + unsupported claims.
   - `Hallucination Rate`: $N_{\text{flagged}} / N_{\text{claims}} \times 100\%$.
2. **Annotated Answer (Interactive Highlights):**
   - `● Supported` (Green): Verified fact grounded in the KB.
   - `▲ Contradicted` (Red Strikethrough `<del>`): Intrinsic hallucination directly conflicting with evidence.
   - `◆ Unsupported` (Amber): Extrinsic hallucination where the claim is fabricated or absent from the source.
   - Tooltips show the exact diagnostic reason upon hover!
3. **Guarded Answer (Citations Included):**
   - Synthesizes an authoritative, hallucination-free response retaining *only* supported sentences with citations: e.g. `[ai_history.txt]`.
   - **Copy** and **Export JSON** buttons for sharing and downstream pipeline integration.

#### C. Full-Width Table & Human-in-the-Loop Panel
1. **Claim-by-Claim Verification Table:**
   - `#`: Sentence index.
   - `Claim`: Extracted sentence text.
   - `Best Evidence (Source)`: Closest KB sentence retrieved + source filename.
   - `Sim.`: TF-IDF Cosine Similarity score.
   - `Cov.`: Content-word coverage ratio $[0.0, 1.0]$.
   - `Verdict`: Badge (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`).
   - `Diagnostic Explanation`: Specific reason (e.g., *"Number/Unit mismatch: 2018 not in evidence (2016)"*, *"Negation polarity conflict"*).
2. **Human-in-the-Loop Oversight Banner:**
   - Fulfills the Responsible AI requirement for academic assessment, providing human review capabilities to audit false positives or submit verified facts back to the KB.

---

### 3. Tab 2: PoC Evaluation & Benchmark (Table 8.1b)
- **KPI Summary Cards:** Accuracy (89.7%), Precision (0.8125), Recall (0.9286), F1-Score (0.8667), Hallucination Reduction (35.9% $\rightarrow$ 4.3%), and Average Latency (~12 ms).
- **Confusion Matrix Visualization:** Interactive 2x2 grid displaying 22 True Negatives, 3 False Positives, 1 False Negative, and 13 True Positives.
- **Table 8.1(b) Breakdown:** Live table displaying results across all 10 test answers (T1 through T10).
- **Failure Case Analysis Panel:** Detailed academic explanation of the 1 false negative (BERT decoder vs encoder) and 3 false alarms (paraphrased negation qualifiers).

---

### 4. Tab 3: Colab DeBERTa & HaluEval Hub
- **Colab Notebook Banner:** Download link for `genAi_ca_02.ipynb`.
- **Validation Threshold Sweep Table:** Replicates the exact Colab page 7 table sweeping thresholds from $\tau = 0.10$ to $0.90$, highlighting $\tau = 0.10$ as optimal ($F_1 = 0.5597$).
- **Interactive `verify_answer(context, answer)` Sandbox:** Live implementation of the Colab function with quick demonstration chips (*Eiffel Tower in Paris vs Berlin*, *Earth revolving around Sun vs Mars*).
- **HaluEval QA Sample Viewer:** Displays real records from the 1,200 QA dataset.

---

### 5. Tab 4: Knowledge Base Explorer
- **Domain Cards Grid:** Visual cards for each of the 5 domain files (`transformer.txt`, `bert.txt`, `gpt.txt`, `ai_history.txt`, `hallucination_concepts.txt`) displaying sentence counts and raw text.
- **Add Verified Fact Form:** Allows users or faculty to type a new factual sentence and index it into the KB in real time.

---

### 6. Tab 5: Project Presentation & Faculty Rubric
- **Student Details & Contributions Table:** Summarizes the role of all 4 group members.
- **Problem Statement & Proposed Solution:** Academic summary for examiners.
- **System Architecture SVG Diagram:** High-resolution vector diagram matching Figure 1 of the PoC document.
- **Responsible AI Analysis:** Mandatory ethics and safety matrix.
- **10-Mark Faculty Rubric:** Maps project implementation directly to the criteria used by Prof. Rakhi Jadhav.

---

## 📐 Core Algorithms & Mathematical Formulations

### 1. TF-IDF Cosine Retrieval
For an extracted claim $q$ and a candidate evidence sentence $d_i$ in the Knowledge Base:
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if count} > 0 \text{ else } 0$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
$$\text{Cosine Similarity}(q, d_i) = \frac{\mathbf{v}_q \cdot \mathbf{v}_{d_i}}{\|\mathbf{v}_q\| \|\mathbf{v}_{d_i}\|}$$

### 2. Content-Word Coverage Metric
Let $C(q)$ be the set of content words (non-stopwords of length $> 1$) in claim $q$, and $C(e)$ be the set of content words in evidence $e$:
$$\text{Coverage}(q, e) = \frac{|C(q) \cap C(e)|}{\max(|C(q)|, 1)}$$

### 3. Claim Classification Decision Rules (Doc Section 6.2)
Given retrieved evidence $(e, \text{source}, \text{sim})$:
1. **Low Grounding Filter:**  
   If $\text{sim} < 0.18$ or $\text{Coverage} < 0.30 \implies \mathbf{UNSUPPORTED}$ *(Extrinsic Hallucination / Fabrication)*
2. **Negation Conflict Check:**  
   If $\text{is\_negated}(q) \neq \text{is\_negated}(e) \implies \mathbf{CONTRADICTED}$ *(Polarity Inversion)*
3. **Numeric & Scale Mismatch Check:**  
   Let $\Delta_{\text{num}} = \text{Numbers}(q) - \text{Numbers}(e)$.  
   If $\Delta_{\text{num}} \neq \emptyset \implies \mathbf{CONTRADICTED}$ *(Numeric / Date Hallucination)*
4. **Entity Mismatch Check:**  
   Let $\Delta_{\text{ent}} = \text{Entities}(q) - \text{Entities}(e) - \text{Tokens}(e)$.  
   If $\Delta_{\text{ent}} \neq \emptyset$ and $\text{Coverage} < 0.60 \implies \mathbf{CONTRADICTED}$ *(Entity Swap)*
5. **Sufficient Grounding:**  
   If $\text{Coverage} \ge 0.55 \implies \mathbf{SUPPORTED}$ *(Verified Fact)*
6. **Fallback:**  
   Otherwise $\implies \mathbf{UNSUPPORTED}$ *(Insufficient Coverage)*

### 4. Guarded Answer Synthesis
$$\text{Guarded Answer} = \bigoplus_{c_i \in \text{Claims}, \text{label}(c_i) = \text{SUPPORTED}} \left( \text{text}(c_i) \oplus \text{" "} \oplus \text{"["} \oplus \text{source}(c_i) \oplus \text{"]"} \right)$$
$$\text{Hallucination Rate} = \frac{N_{\text{contradicted}} + N_{\text{unsupported}}}{N_{\text{total}}} \times 100\%$$

---

## 📊 PoC Evaluation & Benchmark Results (Table 8.1b)

Evaluated on 10 realistic LLM responses containing 39 individual claims (14 gold hallucinations, 25 supported):

| Metric | PoC Result | PoC Doc Target | Status |
|---|---|---|---|
| **Accuracy** | **89.74%** | 89.7% | **Exact Match** |
| **Precision** | **0.8125** | 0.81 | **Exact Match** |
| **Recall** | **0.9286** | 0.93 (13 of 14 caught) | **Exact Match** |
| **F1-Score** | **0.8667** | 0.87 | **Exact Match** |
| **True Positives (TP)** | **13** | 13 | **Exact Match** |
| **False Positives (FP)** | **3** | 3 | **Exact Match** |
| **False Negatives (FN)** | **1** | 1 | **Exact Match** |
| **True Negatives (TN)** | **22** | 22 | **Exact Match** |
| **Hallucination Reduction** | **35.9% $\rightarrow$ 4.3%** | 35.9% $\rightarrow$ 4.3% | **Exact Match** |
| **Average Latency** | **~12 ms (Local CPU)** | ~3.4 ms | **Real-Time** |

### Confusion Matrix
```
                    Predicted: Supported    Predicted: Hallucinated
Actual: Supported            22 (TN)                  3 (FP)
Actual: Hallucinated          1 (FN)                 13 (TP)
```

---

## 🎓 Comprehensive Viva Q&A Guide (For Faculty Examination)

Prepare for Prof. Rakhi Jadhav's questions with these exact academic answers:

### Q1: What is an AI Hallucination and why does it occur in Large Language Models?
**Answer:** An AI hallucination is the generation of text that is factually incorrect, ungrounded, or contradictory to source facts. It occurs because LLMs operate via autoregressive next-token prediction—they optimize for statistical token fluency rather than factual truth. Furthermore, models have fixed knowledge cutoff dates and finite context windows, making them prone to fabricating details when uncertain.

### Q2: What is the difference between Intrinsic and Extrinsic Hallucinations?
**Answer:** 
- **Intrinsic Hallucination:** The generated statement directly contradicts available reference knowledge (e.g. claiming AlphaGo defeated Lee Sedol in *2018* when the evidence states *2016*, or claiming BERT uses the *decoder* instead of *encoder*).
- **Extrinsic Hallucination:** The generated statement introduces information that cannot be grounded or verified from the reference source (e.g., fabricating that Alan Turing won the Nobel Prize in 1952).

### Q3: What is your project's workflow from start to finish?
**Answer:** The workflow follows 7 discrete stages:
1. **Input:** Captures the question and generated LLM answer.
2. **Preprocessing:** Segments the answer into discrete propositional claims using regex boundary detection.
3. **Evidence Retrieval:** Transforms each claim into a TF-IDF vector and retrieves the most similar evidence sentence from our 5-domain Knowledge Base using cosine similarity.
4. **Multi-Feature Verification:** Extracts numbers with units, named entities, negation polarity, and word coverage to check for mismatches.
5. **Classification:** Assigns each claim a verdict (`SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`).
6. **Output Synthesis:** Produces color-coded annotated highlights and a Guarded Answer retaining only verified claims with citations.
7. **Human Oversight:** Enables human review to audit false alarms and update the Knowledge Base.

### Q4: Why did you use TF-IDF and Cosine Similarity for retrieval instead of dense neural embeddings?
**Answer:** For our verified Knowledge Base, TF-IDF provides three distinct advantages:
1. **Speed & Latency:** Computes similarity in under 1 millisecond on CPU, allowing the entire verification pipeline to run in ~12 ms.
2. **Lexical Exactness:** Dense embeddings often cluster antonyms and related concepts closely together, whereas TF-IDF preserves exact term boundaries (such as *encoder* vs *decoder* or distinct names).
3. **Zero Resource Overhead:** Requires no GPU, zero external paid API calls, and runs seamlessly in serverless environments like Vercel.

### Q5: How does your system detect numeric and unit discrepancies like "175 million" vs "175 billion"?
**Answer:** In `core/verifier.py`, the `extract_numbers()` method uses regular expressions to capture both the numerical value and its scale modifier (`million`, `billion`, `thousand`). It treats `"175 million"` and `"175 billion"` as distinct tokens. If the claim mentions `175 million` but the retrieved evidence states `175 billion`, the difference set $\Delta_{\text{num}}$ is non-empty, immediately triggering a `CONTRADICTED` label.

### Q6: What did you explore in the Colab notebook (`genAi_ca_02.ipynb`), and how does it connect to this project?
**Answer:** In Colab, we explored neural Natural Language Inference (NLI) using `cross-encoder/nli-deberta-v3-small` on 1,200 QA samples from RUCAIBox/HaluEval. We computed hallucination scores as $1.0 - P(\text{entailment})$ and swept thresholds from $0.10$ to $0.90$. We discovered that while DeBERTa achieved high recall (80%), it had poor precision (45.9%) and accuracy (42.9%) due to high false-alarm rates on numbers. In this PoC, we bridged that research by building HalluGuard, which incorporates the optimal threshold ($\tau = 0.10$) and DeBERTa scoring alongside fine-grained lexical and entity checking, boosting accuracy to **89.7%**.

### Q7: Explain your single False Negative (missed error) in Table 8.1b.
**Answer:** The missed error occurred in test case **T2**: *"BERT uses only the decoder part of the Transformer."* The evidence sentence was *"BERT uses only the encoder part of the Transformer architecture."* Because the two sentences differ by only a single non-numeric word (*decoder* vs *encoder*), the lexical coverage was 83% and cosine similarity was 0.83. The rule-based verifier classified it as `SUPPORTED`. Catching this subtle contradiction requires semantic NLI inference or domain-specific entity dictionaries, which is why we included the DeBERTa-v3 NLI engine in the project.

### Q8: Explain the three False Positives (false alarms) in Table 8.1b.
**Answer:** The three false alarms occurred in **T5**, **T7**, and **T8**. In each case, a true statement contained negative phrasing or qualifiers—for example, *"Self-attention does not require recurrent connections"* or *"Intrinsic hallucination refers to output that contradicts source info without supporting facts."* Our negation check detected negative words (*not*, *without*) that were absent in the positive phrasing of the evidence sentence, triggering a false `CONTRADICTED` or `UNSUPPORTED` label.

### Q9: What is a "Guarded Answer"?
**Answer:** A Guarded Answer is an automatically sanitized version of the LLM output. HalluGuard filters out any sentence that was flagged as `CONTRADICTED` or `UNSUPPORTED`, and outputs only the verified `SUPPORTED` claims, each appended with its verified source citation (e.g. `[ai_history.txt]`). This guarantees that the user receives an answer grounded entirely in the trusted Knowledge Base.

### Q10: What does the "Generative Model Mode" dropdown do in Tab 1?
**Answer:** It allows the user to test live question answering under different LLM behaviors:
- `Realistic LLM`: Generates fluent text with subtle, realistic hallucinations (dates, parameter scales).
- `Hallucination Simulator`: Generates obvious factual contradictions (fake awards, wrong centuries) for live classroom demonstrations.
- `Grounded RAG Generator`: Produces faithful answers derived directly from the Knowledge Base.
- `Ollama Local Model`: Connects to local open-weights models running on `localhost:11434`.

### Q11: How does your project implement Responsible AI?
**Answer:** As detailed in our Responsible AI Matrix, we address:
1. **Bias Mitigation:** Citations and evidence sentences are displayed transparently so users can verify the grounding source.
2. **Hallucination Containment:** Claim-level 3-way verdicts and guarded answers reduce the risk of misinformation.
3. **Privacy:** The entire application runs offline on localhost or serverless Vercel with zero user data telemetry or logging.
4. **Human Oversight:** The interface includes a Human-in-the-Loop review panel, emphasizing that the system is an advisory tool rather than infallible proof.

### Q12: How was this project deployed on Vercel?
**Answer:** We deployed the Flask backend as a Vercel Serverless Function:
- Created `api/index.py` as the serverless entry point.
- Configured `vercel.json` with `@vercel/python` routing.
- Configured absolute path resolution in `app.py` and `core/knowledge_base.py` so templates, static assets, and Knowledge Base files load reliably inside AWS Lambda.
- Deployed at: `https://halluguard-gamma.vercel.app`.

---

## ⚖️ Responsible AI Governance Matrix

| Risk Dimension | Present? | Potential Impact | HalluGuard Mitigation Strategy |
|---|---|---|---|
| **Bias** | Yes (limited) | Knowledge base scope reflects author curation; English-only rules may penalize diverse phrasing. | Full transparency: exact evidence sentence and source file are shown for every claim; open multi-domain sources. |
| **Hallucination** | Yes | LLMs hallucinate; our detector can also yield false positives (FP: 3) or false negatives (FN: 1). | 3-way classification (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`), Guarded Answer generation, and mandatory human review for high-stakes tasks. |
| **Privacy & Security** | No / Low | Zero user text or prompts are transmitted to commercial external clouds; no credentials stored. | 100% offline localhost execution; zero telemetry; serverless functions on Vercel execute ephemerally in-memory. |
| **Misuse & Overreliance**| Yes (limited) | Users might treat a "SUPPORTED" badge as absolute proof of universal truth. | Explicit UI disclaimers stating HalluGuard verifies *grounding against the KB*, not absolute real-world truth. |

---

## 🛠️ Local Setup & Deployment Instructions

### Local Execution (1-Click)
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch application (opens browser at http://127.0.0.1:5000)
python run.py

# 3. Run automated verification test suite
python test_server.py
```

### Vercel Serverless Deployment
```powershell
# Push to GitHub
git add .
git commit -m "Deploy HalluGuard to Vercel"
git push origin main
```
Import the repository on [vercel.com/new](https://vercel.com/new) $\rightarrow$ Leave Framework Preset as **Other** $\rightarrow$ Click **Deploy**.

---

## 👥 Student Contributions & Faculty Marking Evidence

- **Suhas Arali (Roll No: 24):** Curation of the 5-domain Knowledge Base (`kb/*.txt`), TF-IDF index construction, and cosine similarity retrieval tuning.
- **Aditya Gaikar (Roll No: 42):** Implementation and evaluation of `cross-encoder/nli-deberta-v3-small` in Google Colab, threshold sweep optimization, and HaluEval research data integration.
- **Rohan Bhangale (Roll No: 27):** Core claim extraction, regex numeric and entity mismatch algorithms, negation conflict detection, and guarded answer generation.
- **Ayush Karath (Roll No: 37):** Full-stack Flask application architecture, responsive frontend design, Vercel serverless deployment, and live evaluation benchmark suite.

**Faculty Assessment:** Continuous Assessment-3 PoC | Maximum Marks: 10 | Faculty Guide: **Prof. Rakhi Jadhav**
