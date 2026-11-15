import time
from typing import Dict, Any, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from .knowledge_base import KnowledgeBase
from .verifier import HallucinationVerifier
from .testset import BENCHMARK_TESTSET


def run_benchmark_evaluation(kb: KnowledgeBase = None) -> Dict[str, Any]:
    """
    Evaluates HalluGuard on the 10 benchmark test cases (39 claims) matching Table 8.1(b).
    """
    if kb is None:
        kb = KnowledgeBase()
    verifier = HallucinationVerifier(kb)

    y_true = []
    y_pred = []
    per_answer_results = []
    detailed_claims = []
    start_total_time = time.perf_counter()

    for item in BENCHMARK_TESTSET:
        t_id = item["id"]
        question = item["question"]
        gold_claims = item["claims"]
        
        t_start = time.perf_counter()
        detected_count = 0
        fp_count = 0
        fn_count = 0
        gold_hal_count = 0

        for c_idx, c_info in enumerate(gold_claims, start=1):
            claim_text = c_info["text"]
            gold_is_hal = c_info["gold_is_hallucination"]
            if gold_is_hal:
                gold_hal_count += 1

            result = verifier.verify_claim(claim_text, index=c_idx)
            pred_is_hal = result.is_hallucination

            y_true.append(1 if gold_is_hal else 0)
            y_pred.append(1 if pred_is_hal else 0)

            if gold_is_hal and pred_is_hal:
                detected_count += 1
            elif not gold_is_hal and pred_is_hal:
                fp_count += 1
            elif gold_is_hal and not pred_is_hal:
                fn_count += 1

            detailed_claims.append({
                "test_id": t_id,
                "question": question,
                "claim": claim_text,
                "gold_label": c_info["gold_label"],
                "gold_is_hal": gold_is_hal,
                "pred_label": result.label,
                "pred_is_hal": pred_is_hal,
                "similarity": result.similarity,
                "coverage": result.coverage,
                "reason": result.reason,
                "evidence": result.evidence,
                "source": result.source,
                "status": "PASS" if gold_is_hal == pred_is_hal else "FAIL"
            })

        t_elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        per_answer_results.append({
            "id": t_id,
            "question": question,
            "total_claims": len(gold_claims),
            "gold_hallucinated": gold_hal_count,
            "detected": detected_count,
            "false_alarms": fp_count,
            "false_negatives": fn_count,
            "elapsed_ms": t_elapsed
        })

    total_time_ms = round((time.perf_counter() - start_total_time) * 1000, 2)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    raw_hal_rate = round((sum(y_true) / len(y_true)) * 100, 1)
    kept_claims = len(y_pred) - sum(y_pred)
    residual_hal_rate = round((fn / max(kept_claims, 1)) * 100, 1)

    errors = [c for c in detailed_claims if c["status"] == "FAIL"]

    return {
        "summary": {
            "total_answers": len(BENCHMARK_TESTSET),
            "total_claims": len(y_true),
            "gold_hallucinations": int(sum(y_true)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "raw_hallucination_rate_pct": raw_hal_rate,
            "residual_hallucination_rate_pct": residual_hal_rate,
            "avg_ms_per_answer": round(total_time_ms / len(BENCHMARK_TESTSET), 2),
            "total_time_ms": total_time_ms
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "per_answer": per_answer_results,
        "detailed_claims": detailed_claims,
        "errors": errors
    }


if __name__ == "__main__":
    results = run_benchmark_evaluation()
    print("=== SUMMARY METRICS ===")
    for k, v in results["summary"].items():
        print(f"{k}: {v}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"]["matrix"])
    print(f"\nErrors Count: {len(results['errors'])}")

