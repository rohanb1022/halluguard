from app import app
import json

client = app.test_client()

print("1. Testing GET /")
r = client.get("/")
assert r.status_code == 200
print(f"Status: {r.status_code}, HTML length: {len(r.data)}")

print("\n2. Testing GET /api/presets")
r = client.get("/api/presets")
assert r.status_code == 200
presets = r.get_json()["presets"]
print(f"Presets count: {len(presets)}")
assert len(presets) == 10

print("\n3. Testing POST /api/verify with Sample T4")
sample_t4 = presets[3]  # T4
res = client.post("/api/verify", data=json.dumps({
    "question": sample_t4["question"],
    "answer": sample_t4["answer"],
    "engine": "lexical"
}), content_type="application/json")
assert res.status_code == 200
data = res.get_json()
print("Total claims:", data["total_claims"])
print("Flagged claims:", data["flagged_claims"])
print("Hallucination rate:", data["hallucination_rate_pct"], "%")
print("Guarded answer:", data["guarded_answer"])
assert data["total_claims"] == 5
assert data["flagged_claims"] == 2

print("\n4. Testing GET /api/benchmark")
r = client.get("/api/benchmark")
assert r.status_code == 200
bm = r.get_json()
s = bm["summary"]
print(f"Benchmark Accuracy: {s['accuracy'] * 100}%")
print(f"Benchmark Precision: {s['precision']}")
print(f"Benchmark Recall: {s['recall']}")
print(f"Benchmark F1: {s['f1_score']}")
print(f"TP: {s['tp']}, FP: {s['fp']}, FN: {s['fn']}, TN: {s['tn']}")
assert s["tp"] == 13
assert s["fp"] == 3
assert s["fn"] == 1
assert s["tn"] == 22

print("\n5. Testing Colab verify_answer (Eiffel Tower)")
res = client.post("/api/colab/verify_answer", data=json.dumps({
    "context": "The Eiffel Tower is located in Paris, France.",
    "answer": "The Eiffel Tower is in Berlin."
}), content_type="application/json")
assert res.status_code == 200
cdata = res.get_json()
print("Eiffel Tower in Berlin prediction:", cdata["prediction"])
assert cdata["is_hallucination"] is True

res2 = client.post("/api/colab/verify_answer", data=json.dumps({
    "context": "The Eiffel Tower is located in Paris, France.",
    "answer": "The Eiffel Tower is in Paris."
}), content_type="application/json")
assert res2.status_code == 200
cdata2 = res2.get_json()
print("Eiffel Tower in Paris prediction:", cdata2["prediction"])
assert cdata2["is_hallucination"] is False

print("\n6. Testing Knowledge Base API")
res_kb = client.get("/api/kb")
assert res_kb.status_code == 200
kb_data = res_kb.get_json()
print(f"KB Total sentences: {kb_data['stats']['total_sentences']}")
print(f"KB Files: {kb_data['stats']['files']}")

print("\n7. Testing Health Endpoint")
res_h = client.get("/api/health")
assert res_h.status_code == 200
print("Health:", res_h.get_json())

print("\n8. Testing Live AI Generation Endpoint (POST /api/generate)")
gen_res = client.post("/api/generate", data=json.dumps({
    "question": "Give key milestones in AI.",
    "mode": "realistic"
}), content_type="application/json")
assert gen_res.status_code == 200
gdata = gen_res.get_json()
print("Generated Answer preview:", gdata["answer"][:90], "...")

print("\n9. Testing Ask AI & Guard Live Workflow (POST /api/ask_and_guard)")
aag_res = client.post("/api/ask_and_guard", data=json.dumps({
    "question": "What is BERT and how does it work?",
    "gen_mode": "realistic"
}), content_type="application/json")
assert aag_res.status_code == 200
adata = aag_res.get_json()
print("Total claims extracted:", adata["total_claims"])
print("Supported claims:", adata["supported_claims"])
print("Flagged claims:", adata["flagged_claims"])
print("Hallucination rate:", adata["hallucination_rate_pct"], "%")
print("Guarded output:", adata["guarded_answer"])
assert adata["total_claims"] > 0

print("\n>>> ALL TESTS PASSED SUCCESSFULLY 100%! <<<")
