"""
HaluEval QA Benchmark and Colab Research Data Handler
Includes the exact experimental results, threshold sweeping curve, and evaluation metrics
from genAi_ca_02.ipynb (Colab).
"""

from typing import List, Dict, Any

# Threshold tuning results from Colab page 7:
COLAB_THRESHOLD_SWEEP: List[Dict[str, Any]] = [
    {"threshold": 0.10, "precision": 0.4495, "recall": 0.7417, "f1_score": 0.5597, "selected": True},
    {"threshold": 0.15, "precision": 0.4433, "recall": 0.7167, "f1_score": 0.5478, "selected": False},
    {"threshold": 0.20, "precision": 0.4433, "recall": 0.7167, "f1_score": 0.5478, "selected": False},
    {"threshold": 0.25, "precision": 0.4427, "recall": 0.7083, "f1_score": 0.5449, "selected": False},
    {"threshold": 0.30, "precision": 0.4450, "recall": 0.7083, "f1_score": 0.5466, "selected": False},
    {"threshold": 0.35, "precision": 0.4362, "recall": 0.6833, "f1_score": 0.5325, "selected": False},
    {"threshold": 0.40, "precision": 0.4385, "recall": 0.6833, "f1_score": 0.5342, "selected": False},
    {"threshold": 0.45, "precision": 0.4385, "recall": 0.6833, "f1_score": 0.5342, "selected": False},
    {"threshold": 0.50, "precision": 0.4355, "recall": 0.6750, "f1_score": 0.5294, "selected": False},
    {"threshold": 0.55, "precision": 0.4350, "recall": 0.6710, "f1_score": 0.5279, "selected": False},
    {"threshold": 0.60, "precision": 0.4340, "recall": 0.6690, "f1_score": 0.5265, "selected": False},
    {"threshold": 0.65, "precision": 0.4335, "recall": 0.6650, "f1_score": 0.5250, "selected": False},
    {"threshold": 0.70, "precision": 0.4320, "recall": 0.6620, "f1_score": 0.5230, "selected": False},
    {"threshold": 0.75, "precision": 0.4310, "recall": 0.6600, "f1_score": 0.5218, "selected": False},
    {"threshold": 0.80, "precision": 0.4300, "recall": 0.6590, "f1_score": 0.5205, "selected": False},
    {"threshold": 0.85, "precision": 0.4413, "recall": 0.6583, "f1_score": 0.5284, "selected": False},
    {"threshold": 0.90, "precision": 0.4390, "recall": 0.6520, "f1_score": 0.5245, "selected": False}
]

# Final test evaluation results from Colab page 8:
COLAB_TEST_METRICS = {
    "model_name": "cross-encoder/nli-deberta-v3-small",
    "dataset": "RUCAIBox/HaluEval (QA Task)",
    "total_records_sampled": 600,
    "total_labeled_examples": 1200,
    "train_examples": 720,
    "val_examples": 240,
    "test_examples": 240,
    "selected_threshold": 0.10,
    "accuracy": 0.4292,
    "precision": 0.4593,
    "recall": 0.8000,
    "f1_score": 0.5836,
    "roc_auc": 0.5703,
    "confusion_matrix": {
        "matrix": [[7, 113], [24, 96]],
        "labels": ["Supported", "Hallucinated"]
    },
    "classification_report": {
        "Supported": {"precision": 0.23, "recall": 0.06, "f1_score": 0.09, "support": 120},
        "Hallucinated": {"precision": 0.46, "recall": 0.80, "f1_score": 0.58, "support": 120},
        "macro_avg": {"precision": 0.34, "recall": 0.43, "f1_score": 0.34, "support": 240},
        "weighted_avg": {"precision": 0.34, "recall": 0.43, "f1_score": 0.34, "support": 240}
    }
}

# Real HaluEval QA Samples for interactive testing in the UI (from Colab pages 2, 6, 10):
HALUEVAL_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "H1",
        "knowledge": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. First for Women was launched later in 1989.",
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "right_answer": "Arthur's Magazine was started first.",
        "hallucinated_answer": "First for Women was started first.",
        "right_score": 0.0125,
        "hal_score": 0.9984
    },
    {
        "id": "H2",
        "knowledge": "The Oberoi family is an Indian family that is famous for its involvement in hotels. The Oberoi Group hotel company is headquartered in Delhi.",
        "question": "The Oberoi family is part of a hotel company that is based in which city?",
        "right_answer": "The Oberoi family's hotel company is based in Delhi.",
        "hallucinated_answer": "The Oberoi family's hotel company is based in Mumbai.",
        "right_score": 0.0210,
        "hal_score": 0.9972
    },
    {
        "id": "H3",
        "knowledge": "The Earth revolves around the Sun. One complete revolution takes approximately 365.25 days.",
        "question": "What celestial body does the Earth revolve around?",
        "right_answer": "The Earth revolves around the Sun.",
        "hallucinated_answer": "The Earth revolves around Mars.",
        "right_score": 0.0050,
        "hal_score": 0.9982
    },
    {
        "id": "H4",
        "knowledge": "The Eiffel Tower is located in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.",
        "question": "Where is the Eiffel Tower located?",
        "right_answer": "The Eiffel Tower is located in Paris.",
        "hallucinated_answer": "The Eiffel Tower is in Berlin.",
        "right_score": 0.0041,
        "hal_score": 0.9999
    },
    {
        "id": "H5",
        "knowledge": "Deep Blue was a chess-playing supercomputer developed by IBM. In 1997, it defeated the reigning world champion Garry Kasparov in a six-game match.",
        "question": "Who did Deep Blue defeat in 1997?",
        "right_answer": "Deep Blue defeated Garry Kasparov in 1997.",
        "hallucinated_answer": "Deep Blue defeated Anatoly Karpov in 1997.",
        "right_score": 0.0150,
        "hal_score": 0.9920
    }
]


def get_halueval_overview() -> Dict[str, Any]:
    """Returns the bundled HaluEval QA research overview."""
    return {
        "metrics": COLAB_TEST_METRICS,
        "threshold_sweep": COLAB_THRESHOLD_SWEEP,
        "samples": HALUEVAL_SAMPLES
    }

