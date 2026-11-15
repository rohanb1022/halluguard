import os
import time
from flask import Flask, render_template, request, jsonify
from core.knowledge_base import KnowledgeBase
from core.verifier import HallucinationVerifier
from core.nli_engine import NLICrossEncoderEngine
from core.testset import BENCHMARK_TESTSET
from core.evaluate import run_benchmark_evaluation
from core.halueval_bench import get_halueval_overview
from core.generator import LLMGenerator

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Initialize singletons
kb = KnowledgeBase(kb_dir=os.path.join(BASE_DIR, "kb"))
lexical_verifier = HallucinationVerifier(kb)
nli_engine = NLICrossEncoderEngine()
generator = LLMGenerator(kb)


@app.route("/")
def index():
    """Renders the main single-page application."""
    kb_stats = kb.get_stats()
    return render_template(
        "index.html",
        kb_stats=kb_stats,
        default_model=NLICrossEncoderEngine.MODEL_NAME
    )


@app.route("/api/generate", methods=["POST"])
def generate_endpoint():
    """
    Generates an LLM answer to any user-typed question in real time.
    Supports realistic, hallucinatory (demo mode), grounded, or Ollama local models.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "realistic").lower()
    model_name = data.get("model_name", "default")

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    gen_result = generator.generate_answer(question, mode=mode, model_name=model_name)
    return jsonify(gen_result)


@app.route("/api/ask_and_guard", methods=["POST"])
def ask_and_guard_endpoint():
    """
    End-to-End Live Workflow:
    1. Generates an answer to any custom question using the selected LLM mode.
    2. Runs HalluGuard verification on the generated answer.
    3. Returns both the generated response and the claim-by-claim hallucination report.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    gen_mode = data.get("gen_mode", "realistic").lower()
    model_name = data.get("model_name", "default")
    engine_type = data.get("engine", "lexical").lower()
    use_custom_ctx = bool(data.get("use_custom_context", False))
    custom_ctx = data.get("custom_context", "").strip()
    threshold = float(data.get("threshold", 0.10))

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    # Step 1: Generate answer
    gen_result = generator.generate_answer(question, mode=gen_mode, model_name=model_name)
    generated_answer = gen_result["answer"]

    # Step 2: Verify using the verification pipeline
    active_kb = kb
    if use_custom_ctx and custom_ctx:
        active_kb = KnowledgeBase()
        active_kb.load_custom_context(custom_ctx, source_name="custom_user_doc.txt")
        verifier_to_use = HallucinationVerifier(active_kb)
    else:
        verifier_to_use = lexical_verifier

    start_verify_time = time.perf_counter()

    if engine_type == "nli":
        sentences = verifier_to_use.split_sentences(generated_answer)
        claims = []
        for idx, sentence in enumerate(sentences, start=1):
            ev, src, sim = active_kb.retrieve(sentence)
            nli_res = nli_engine.verify_answer(ev, sentence, threshold=threshold)
            score = nli_res["hallucination_score"]
            is_hal = nli_res["is_hallucination"]

            if not is_hal:
                label = "SUPPORTED"
                reason = f"NLI Entailment high ({round(1.0 - score, 4)}). Verified against {src}."
            elif score > 0.85:
                label = "CONTRADICTED"
                reason = f"NLI Contradiction detected (Hallucination score: {score}). Mismatches evidence."
            else:
                label = "UNSUPPORTED"
                reason = f"NLI Low entailment / Neutral (Hallucination score: {score}). Insufficient backing."

            claims.append({
                "index": idx,
                "text": sentence,
                "evidence": ev,
                "source": src,
                "similarity": sim,
                "coverage": round(1.0 - score, 3),
                "label": label,
                "reason": reason,
                "mismatches": [f"NLI Hallucination Score: {score}"] if is_hal else [],
                "is_hallucination": is_hal
            })

        total = len(claims)
        supported = sum(1 for c in claims if c["label"] == "SUPPORTED")
        contradicted = sum(1 for c in claims if c["label"] == "CONTRADICTED")
        unsupported = sum(1 for c in claims if c["label"] == "UNSUPPORTED")
        flagged = contradicted + unsupported
        rate_pct = round((flagged / max(total, 1)) * 100, 1)

        guarded_parts = [f"{c['text']} [{c['source']}]" for c in claims if c["label"] == "SUPPORTED"]
        guarded_answer = " ".join(guarded_parts) if guarded_parts else "[No supported claims verified in this answer.]"

        annotated_spans = []
        for c in claims:
            if c["label"] == "SUPPORTED":
                span = f'<span class="claim-badge badge-supported" data-id="{c["index"]}" title="SUPPORTED: {c["reason"]}">{c["text"]}</span>'
            elif c["label"] == "CONTRADICTED":
                span = f'<span class="claim-badge badge-contradicted" data-id="{c["index"]}" title="CONTRADICTED: {c["reason"]}"><del>{c["text"]}</del></span>'
            else:
                span = f'<span class="claim-badge badge-unsupported" data-id="{c["index"]}" title="UNSUPPORTED: {c["reason"]}">{c["text"]}</span>'
            annotated_spans.append(span)

        elapsed_ms = round((time.perf_counter() - start_verify_time) * 1000, 2)
        report_dict = {
            "total_claims": total,
            "supported_claims": supported,
            "contradicted_claims": contradicted,
            "unsupported_claims": unsupported,
            "flagged_claims": flagged,
            "hallucination_rate_pct": rate_pct,
            "response_time_ms": elapsed_ms,
            "claims": claims,
            "annotated_html": " ".join(annotated_spans),
            "guarded_answer": guarded_answer,
            "engine": "DeBERTa-v3 NLI Cross-Encoder"
        }
    else:
        report = verifier_to_use.verify_answer(generated_answer)
        report_dict = report.to_dict()
        report_dict["engine"] = "Fast Lexical & Rule-Based Verifier"

    report_dict["question"] = question
    report_dict["generated_answer"] = generated_answer
    report_dict["generator_info"] = gen_result
    return jsonify(report_dict)


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Returns the 10 benchmark preset samples from Table 8.1b."""
    presets = []
    for item in BENCHMARK_TESTSET:
        presets.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "claim_count": len(item["claims"]),
            "expected_hallucinations": sum(1 for c in item["claims"] if c["gold_is_hallucination"])
        })
    return jsonify({"presets": presets})


@app.route("/api/verify", methods=["POST"])
def verify():
    """
    Main verification endpoint.
    Verifies input answer sentence-by-sentence using Lexical, NLI, or Hybrid engine.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    engine_type = data.get("engine", "lexical").lower()  # "lexical", "nli", or "hybrid"
    use_custom_ctx = bool(data.get("use_custom_context", False))
    custom_ctx = data.get("custom_context", "").strip()
    threshold = float(data.get("threshold", 0.10))

    if not answer:
        return jsonify({"error": "Answer text is required."}), 400

    # Handle custom context vs default KB
    active_kb = kb
    if use_custom_ctx and custom_ctx:
        active_kb = KnowledgeBase()
        active_kb.load_custom_context(custom_ctx, source_name="custom_user_doc.txt")
        verifier_to_use = HallucinationVerifier(active_kb)
    else:
        verifier_to_use = lexical_verifier

    start_time = time.perf_counter()

    if engine_type == "nli":
        # NLI Cross-Encoder Engine (Colab DeBERTa approach)
        sentences = verifier_to_use.split_sentences(answer)
        claims = []
        for idx, sentence in enumerate(sentences, start=1):
            ev, src, sim = active_kb.retrieve(sentence)
            nli_res = nli_engine.verify_answer(ev, sentence, threshold=threshold)
            score = nli_res["hallucination_score"]
            is_hal = nli_res["is_hallucination"]
            
            if not is_hal:
                label = "SUPPORTED"
                reason = f"NLI Entailment high ({round(1.0 - score, 4)}). Verified against {src}."
            elif score > 0.85:
                label = "CONTRADICTED"
                reason = f"NLI Contradiction detected (Hallucination score: {score}). Mismatches evidence."
            else:
                label = "UNSUPPORTED"
                reason = f"NLI Low entailment / Neutral (Hallucination score: {score}). Insufficient backing."

            claims.append({
                "index": idx,
                "text": sentence,
                "evidence": ev,
                "source": src,
                "similarity": sim,
                "coverage": round(1.0 - score, 3),
                "label": label,
                "reason": reason,
                "mismatches": [f"NLI Hallucination Score: {score}"] if is_hal else [],
                "is_hallucination": is_hal
            })

        total = len(claims)
        supported = sum(1 for c in claims if c["label"] == "SUPPORTED")
        contradicted = sum(1 for c in claims if c["label"] == "CONTRADICTED")
        unsupported = sum(1 for c in claims if c["label"] == "UNSUPPORTED")
        flagged = contradicted + unsupported
        rate_pct = round((flagged / max(total, 1)) * 100, 1)

        guarded_parts = [f"{c['text']} [{c['source']}]" for c in claims if c["label"] == "SUPPORTED"]
        guarded_answer = " ".join(guarded_parts) if guarded_parts else "[No supported claims verified in this answer.]"

        annotated_spans = []
        for c in claims:
            if c["label"] == "SUPPORTED":
                span = f'<span class="claim-badge badge-supported" data-id="{c["index"]}" title="SUPPORTED: {c["reason"]}">{c["text"]}</span>'
            elif c["label"] == "CONTRADICTED":
                span = f'<span class="claim-badge badge-contradicted" data-id="{c["index"]}" title="CONTRADICTED: {c["reason"]}"><del>{c["text"]}</del></span>'
            else:
                span = f'<span class="claim-badge badge-unsupported" data-id="{c["index"]}" title="UNSUPPORTED: {c["reason"]}">{c["text"]}</span>'
            annotated_spans.append(span)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        report_dict = {
            "total_claims": total,
            "supported_claims": supported,
            "contradicted_claims": contradicted,
            "unsupported_claims": unsupported,
            "flagged_claims": flagged,
            "hallucination_rate_pct": rate_pct,
            "response_time_ms": elapsed_ms,
            "claims": claims,
            "annotated_html": " ".join(annotated_spans),
            "guarded_answer": guarded_answer,
            "engine": "DeBERTa-v3 NLI Cross-Encoder"
        }

    elif engine_type == "hybrid":
        # Hybrid Ensemble: Lexical precision for numbers/dates + NLI semantic verification
        base_report = verifier_to_use.verify_answer(answer).to_dict()
        enhanced_claims = []
        for c in base_report["claims"]:
            nli_res = nli_engine.verify_answer(c["evidence"], c["text"], threshold=threshold)
            nli_score = nli_res["hallucination_score"]
            # If lexical said supported but NLI shows extreme contradiction, flag it!
            if c["label"] == "SUPPORTED" and nli_score > 0.85:
                c["label"] = "CONTRADICTED"
                c["is_hallucination"] = True
                c["reason"] = f"Semantic contradiction flagged by DeBERTa NLI (score {nli_score})."
                c["mismatches"].append("Semantic NLI contradiction")
            enhanced_claims.append(c)

        total = len(enhanced_claims)
        supported = sum(1 for c in enhanced_claims if c["label"] == "SUPPORTED")
        contradicted = sum(1 for c in enhanced_claims if c["label"] == "CONTRADICTED")
        unsupported = sum(1 for c in enhanced_claims if c["label"] == "UNSUPPORTED")
        flagged = contradicted + unsupported

        base_report["claims"] = enhanced_claims
        base_report["supported_claims"] = supported
        base_report["contradicted_claims"] = contradicted
        base_report["unsupported_claims"] = unsupported
        base_report["flagged_claims"] = flagged
        base_report["hallucination_rate_pct"] = round((flagged / max(total, 1)) * 100, 1)
        base_report["engine"] = "Hybrid Ensemble (Lexical Rules + DeBERTa NLI)"
        base_report["response_time_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        report_dict = base_report

    else:
        # Default Lexical & Rule-Based Verifier (Doc Section 6.2)
        report = verifier_to_use.verify_answer(answer)
        report_dict = report.to_dict()
        report_dict["engine"] = "Fast Lexical & Rule-Based Verifier"

    report_dict["question"] = question
    return jsonify(report_dict)


@app.route("/api/colab/verify_answer", methods=["POST"])
def colab_verify_endpoint():
    """
    Executes the exact verify_answer(context, answer) function from Colab page 10.
    """
    data = request.get_json(force=True, silent=True) or {}
    context = data.get("context", "").strip()
    answer = data.get("answer", "").strip()
    threshold = float(data.get("threshold", 0.10))

    if not context or not answer:
        return jsonify({"error": "Both context and answer are required."}), 400

    result = nli_engine.verify_answer(context, answer, threshold=threshold)
    return jsonify(result)


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    """Runs the 10-answer, 39-claim benchmark and returns evaluation metrics (Table 8.1b)."""
    results = run_benchmark_evaluation(kb)
    return jsonify(results)


@app.route("/api/halueval", methods=["GET"])
def get_halueval():
    """Returns the bundled HaluEval QA research overview and sample items from Colab."""
    overview = get_halueval_overview()
    return jsonify(overview)


@app.route("/api/kb", methods=["GET"])
def get_kb_details():
    """Returns the knowledge base contents, files, and stats."""
    files_data = []
    for fname, content in kb.file_contents.items():
        sentences = [s for s, src in zip(kb.sentences, kb.sources) if src == fname]
        files_data.append({
            "filename": fname,
            "sentence_count": len(sentences),
            "sentences": sentences,
            "raw_content": content
        })
    return jsonify({
        "stats": kb.get_stats(),
        "files": files_data
    })


@app.route("/api/kb/add", methods=["POST"])
def add_kb_fact():
    """Adds a new fact sentence to the knowledge base."""
    data = request.get_json(force=True, silent=True) or {}
    fact = data.get("fact", "").strip()
    source_file = data.get("source_file", "custom_facts.txt").strip()

    if not fact:
        return jsonify({"error": "Fact sentence cannot be empty."}), 400

    success = kb.add_fact(fact, source_file=source_file)
    return jsonify({"success": success, "stats": kb.get_stats()})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "app_title": "HalluGuard PoC",
        "kb_sentences": len(kb.sentences),
        "device": nli_engine.device,
        "nli_model": nli_engine.MODEL_NAME
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Starting HalluGuard: Claim-Level Hallucination Guard")
    print(f"Indexed {len(kb.sentences)} knowledge base facts across {len(kb.file_contents)} domains.")
    print("Open http://127.0.0.1:5000 in your browser.")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)

