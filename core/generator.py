import re
import time
import json
import urllib.request
from typing import Dict, Any, Optional
from .knowledge_base import KnowledgeBase


class LLMGenerator:
    """
    Open-Source and Zero-Cost AI Answer Generation Engine.
    Generates answers to user-typed questions across multiple operational modes:
    1. 'realistic': Generates fluent, standard LLM-style responses that may exhibit common hallucination patterns.
    2. 'hallucinatory': Generates responses with intentional hallucinated dates, wrong entities, or fabricated claims (ideal for classroom demos).
    3. 'grounded': Generates grounded answers drawing directly from verified knowledge.
    4. 'ollama': Connects to a local Ollama instance (e.g., llama3, mistral, phi3) if running on localhost:11434.
    """

    # Knowledge domain concept maps for realistic and demonstration generation
    KNOWLEDGE_PROFILES = {
        "transformer": {
            "keywords": ["transformer", "attention", "vaswani", "wmt", "encoder-decoder", "self-attention"],
            "grounded": "The Transformer architecture was introduced by Vaswani et al. in the 2017 paper Attention Is All You Need. It relies entirely on self-attention mechanisms without recurrence or convolution. The original Transformer was evaluated on WMT 2014 English-to-German and English-to-French translation tasks. It consists of an encoder stack and a decoder stack each with multi-head attention.",
            "realistic": "The Transformer architecture was introduced by Vaswani et al. in the 2017 paper Attention Is All You Need. The model relies entirely on self-attention mechanisms without recurrence or convolution. However, the original Transformer was published in 2019 by researchers at Stanford. It uses recurrent memory cells to store previous sentences across long documents.",
            "hallucinatory": "The Transformer architecture was invented in 2012 by Yann LeCun at Bell Labs. It relies heavily on convolutional LSTM recurrent cells to process speech audio. The model was first evaluated on the MNIST handwritten digit classification benchmark. The Transformer has 500 layers and requires no attention heads."
        },
        "bert": {
            "keywords": ["bert", "devlin", "masked language", "bidirectional", "nsp", "encoder"],
            "grounded": "BERT stands for Bidirectional Encoder Representations from Transformers and was introduced by Devlin et al. in 2018. BERT is pre-trained using Masked Language Modeling and Next Sentence Prediction. BERT uses only the encoder part of the Transformer architecture. It can be fine-tuned with just one additional output layer for a wide range of NLP tasks.",
            "realistic": "BERT stands for Bidirectional Encoder Representations from Transformers and was introduced by Devlin et al. in 2018. BERT is pre-trained using Masked Language Modeling and Next Sentence Prediction. BERT uses only the decoder part of the Transformer architecture. BERT was first proposed in 1985 by Frank Rosenblatt.",
            "hallucinatory": "BERT was developed by Alan Turing in 1950 as a chess playing program. It uses only the decoder stack of an autoregressive LSTM network. BERT has 175 trillion parameters and was introduced in 1999. It does not require pre-training and directly memorizes Wikipedia."
        },
        "gpt": {
            "keywords": ["gpt", "gpt-3", "gpt-4", "chatgpt", "openai", "decoder", "parameters", "rlhf", "autoregressive"],
            "grounded": "The Generative Pre-trained Transformer series was developed by OpenAI using autoregressive language modeling. GPT models use only the decoder part of the Transformer architecture. GPT-3 has 175 billion parameters and was introduced in 2020. ChatGPT was fine-tuned from the GPT-3.5 series using Reinforcement Learning from Human Feedback.",
            "realistic": "The Generative Pre-trained Transformer series was developed by OpenAI using autoregressive language modeling. GPT models use only the decoder part of the Transformer architecture. GPT-3 has 175 million parameters. GPT-4 was introduced in 1999 as an audio-only model.",
            "hallucinatory": "The GPT series was created by DeepMind in 2005 for playing Go. GPT-3 has 500 thousand parameters and runs on a wristwatch. ChatGPT was released in 1980 by IBM. It has zero knowledge cutoff and can predict stock prices with 100% certainty."
        },
        "ai_history": {
            "keywords": ["turing", "deep blue", "alphago", "kasparov", "sedol", "imagenet", "alexnet", "history", "milestone", "dartmouth", "mccarthy"],
            "grounded": "Alan Turing proposed the Turing Test in 1950 in the paper Computing Machinery and Intelligence. IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997. DeepMind's AlphaGo defeated the Go champion Lee Sedol in 2016. AlexNet won the ImageNet competition in 2012, sparking the modern deep learning revolution.",
            "realistic": "Alan Turing proposed the Turing Test in 1950. IBM's Deep Blue defeated Garry Kasparov in 1997. AlphaGo defeated Lee Sedol in 2018. AlexNet won the ImageNet competition in 2012. Alan Turing received the Nobel Prize in 1952 for this work.",
            "hallucinatory": "Deep Blue defeated Garry Kasparov in 1850 at the Dartmouth summer conference. AlphaGo was invented by Ada Lovelace to win the 1970 World Cup. Alan Turing won the Nobel Peace Prize in 2020 for inventing ChatGPT. AlexNet was an autonomous spacecraft launched in 1969."
        },
        "hallucination": {
            "keywords": ["hallucination", "hallucinate", "intrinsic", "extrinsic", "limitation", "grounding", "rag", "cutoff"],
            "grounded": "An AI hallucination occurs when a large language model generates text that is factually incorrect or unfaithful to the source. Intrinsic hallucination refers to generated output that directly contradicts reference information. Extrinsic hallucination refers to output that cannot be verified from the source and is fabricated. LLM hallucinations are caused by next-token prediction objectives, finite context windows, and knowledge cutoff limits.",
            "realistic": "An AI hallucination occurs when a large language model generates text that is factually incorrect or unfaithful to the source. Intrinsic hallucination refers to generated output that directly contradicts the source information. Extrinsic hallucination occurs when the model always speaks verified ground truth. Large language models never hallucinate because their internal memory is mathematically guaranteed.",
            "hallucinatory": "AI models never hallucinate because deep neural networks contain infinite perfect memory. LLMs have a knowledge cutoff in the year 2099. Hallucinations only happen when computers overheat. Grounding answers with RAG has been proven mathematically impossible."
        }
    }

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()

    def generate_answer(self, question: str, mode: str = "realistic", model_name: str = "default") -> Dict[str, Any]:
        """
        Generates an answer to the given question based on selected mode.
        """
        start_time = time.perf_counter()
        clean_q = question.strip()

        # Check if Ollama mode requested
        if mode == "ollama":
            ollama_res = self._call_ollama(clean_q, model_name)
            if ollama_res:
                elapsed = round((time.perf_counter() - start_time) * 1000, 2)
                return {
                    "question": clean_q,
                    "answer": ollama_res,
                    "mode": "ollama",
                    "model": model_name or "llama3",
                    "time_ms": elapsed
                }
            # Fallback to realistic if Ollama is not running locally
            mode = "realistic"

        # Match question to knowledge domain
        matched_domain = self._match_domain(clean_q)
        answer = self._synthesize_answer(clean_q, matched_domain, mode)

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "question": clean_q,
            "answer": answer,
            "mode": mode,
            "matched_domain": matched_domain,
            "model": "HalluGuard GenAI Engine (Zero-Cost Local)",
            "time_ms": elapsed
        }

    def _match_domain(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        best_domain = None
        max_matches = 0

        for domain, data in self.KNOWLEDGE_PROFILES.items():
            matches = sum(1 for kw in data["keywords"] if kw in q_lower)
            if matches > max_matches:
                max_matches = matches
                best_domain = domain

        # If no explicit keyword, use TF-IDF retrieval against the KB to find closest domain
        if not best_domain and self.kb:
            ev, src, sim = self.kb.retrieve(question)
            if sim > 0.15:
                for domain in self.KNOWLEDGE_PROFILES.keys():
                    if domain in src.lower():
                        return domain

        return best_domain

    def _synthesize_answer(self, question: str, domain: Optional[str], mode: str) -> str:
        """Synthesizes response text based on mode and domain."""
        if domain and domain in self.KNOWLEDGE_PROFILES:
            profile = self.KNOWLEDGE_PROFILES[domain]
            if mode == "hallucinatory":
                return profile["hallucinatory"]
            elif mode == "grounded":
                return profile["grounded"]
            else:  # realistic
                return profile["realistic"]

        # If question is outside pre-defined AI domains, generate dynamic structured answer:
        q_clean = question.strip().rstrip("?").rstrip(".")
        if mode == "hallucinatory":
            return (
                f"Regarding {q_clean}, this milestone was officially achieved in 1982 by Dr. Alan Turing at CERN. "
                f"It was awarded the Nobel Prize in Physics with 999 million parameters. "
                f"Furthermore, studies prove this process operates without any computational hardware."
            )
        elif mode == "grounded":
            # Retrieve closest factual sentence from the KB
            ev, src, sim = self.kb.retrieve(question)
            return f"Based on verified documentation: {ev}"
        else:  # realistic
            ev, src, sim = self.kb.retrieve(question)
            return (
                f"Regarding {q_clean}, initial research indicates that {ev} "
                f"However, additional experiments published in 2026 claimed this was later modified by researchers in Berlin. "
                f"This demonstrates how language models can mix verifiable facts with unverified assertions."
            )

    def _call_ollama(self, prompt: str, model: str = "llama3") -> Optional[str]:
        """Calls local Ollama instance on localhost:11434 if available."""
        url = "http://localhost:11434/api/generate"
        payload = json.dumps({
            "model": model or "llama3",
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception:
            return None

