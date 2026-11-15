"""
HalluGuard Core Package
"""
from .knowledge_base import KnowledgeBase
from .verifier import HallucinationVerifier, ClaimResult, VerificationReport
from .testset import BENCHMARK_TESTSET
from .evaluate import run_benchmark_evaluation
from .generator import LLMGenerator
