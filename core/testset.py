"""
Benchmark Test Set (Table 8.1b of PoC Document)
Consists of 10 hand-labelled LLM-style answers with 39 individual claims.
14 claims are gold-labelled as hallucinations (contradictions, wrong numbers/dates, fabricated details).
25 claims are gold-labelled as supported.
"""

from typing import List, Dict, Any

BENCHMARK_TESTSET: List[Dict[str, Any]] = [
    {
        "id": "T1",
        "question": "Tell me about the Transformer.",
        "answer": "The Transformer architecture was introduced by Vaswani et al. in the 2017 paper Attention Is All You Need. The Transformer relies entirely on self-attention mechanisms without recurrence or convolution. The original Transformer was published in 2025 by Stanford researchers. The Transformer uses recurrent memory cells to store previous sentences.",
        "claims": [
            {
                "text": "The Transformer architecture was introduced by Vaswani et al. in the 2017 paper Attention Is All You Need.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "The Transformer relies entirely on self-attention mechanisms without recurrence or convolution.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "The original Transformer was published in 2025 by Stanford researchers.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            },
            {
                "text": "The Transformer uses recurrent memory cells to store previous sentences.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T2",
        "question": "What is BERT?",
        "answer": "BERT stands for Bidirectional Encoder Representations from Transformers and was introduced by Devlin et al. in 2018. BERT is pre-trained using Masked Language Modeling and Next Sentence Prediction. BERT uses only the decoder part of the Transformer architecture. BERT was first proposed in 1985 by Frank Rosenblatt.",
        "claims": [
            {
                "text": "BERT stands for Bidirectional Encoder Representations from Transformers and was introduced by Devlin et al. in 2018.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "BERT is pre-trained using Masked Language Modeling and Next Sentence Prediction.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "BERT uses only the decoder part of the Transformer architecture.",
                "gold_label": "CONTRADICTED",  # Note: Lexical model misses this (decoder vs encoder) -> False Negative
                "gold_is_hallucination": True
            },
            {
                "text": "BERT was first proposed in 1985 by Frank Rosenblatt.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T3",
        "question": "Describe GPT-3 and ChatGPT.",
        "answer": "The Generative Pre-trained Transformer series was developed by OpenAI using autoregressive language modeling. GPT models use only the decoder part of the Transformer architecture. ChatGPT was fine-tuned from the GPT-3.5 series using Reinforcement Learning from Human Feedback. ChatGPT was released by DeepMind in 2005 for chess tournaments.",
        "claims": [
            {
                "text": "The Generative Pre-trained Transformer series was developed by OpenAI using autoregressive language modeling.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "GPT models use only the decoder part of the Transformer architecture.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "ChatGPT was fine-tuned from the GPT-3.5 series using Reinforcement Learning from Human Feedback.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "ChatGPT was released by DeepMind in 2005 for chess tournaments.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T4",
        "question": "Give key milestones in AI.",
        "answer": "Alan Turing proposed the Turing Test in 1950. Deep Blue defeated Garry Kasparov in 1997. AlphaGo defeated Lee Sedol in 2018. AlexNet won the ImageNet competition in 2012. Alan Turing received the Nobel Prize in 1952 for this work.",
        "claims": [
            {
                "text": "Alan Turing proposed the Turing Test in 1950.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "Deep Blue defeated Garry Kasparov in 1997.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "AlphaGo defeated Lee Sedol in 2018.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            },
            {
                "text": "AlexNet won the ImageNet competition in 2012.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "Alan Turing received the Nobel Prize in 1952 for this work.",
                "gold_label": "UNSUPPORTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T5",
        "question": "What is an AI hallucination?",
        "answer": "An AI hallucination occurs when a large language model generates text that is factually incorrect or unfaithful to the source. Intrinsic hallucination refers to generated output that directly contradicts the source information without supporting facts. Extrinsic hallucination occurs when the model always speaks 100% verified ground truth. Large language models never hallucinate because their internal memory is mathematically guaranteed.",
        "claims": [
            {
                "text": "An AI hallucination occurs when a large language model generates text that is factually incorrect or unfaithful to the source.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "Intrinsic hallucination refers to generated output that directly contradicts the source information without supporting facts.",
                "gold_label": "SUPPORTED",  # Note: contains "without", triggers negation check -> False Positive
                "gold_is_hallucination": False
            },
            {
                "text": "Extrinsic hallucination occurs when the model always speaks 100% verified ground truth.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            },
            {
                "text": "Large language models never hallucinate because their internal memory is mathematically guaranteed.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T6",
        "question": "Where was the Transformer first tested?",
        "answer": "The original Transformer was evaluated on WMT 2014 English-to-German and English-to-French translation tasks. The original Transformer was first deployed in 1995 on autonomous vehicles.",
        "claims": [
            {
                "text": "The original Transformer was evaluated on WMT 2014 English-to-German and English-to-French translation tasks.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "The original Transformer was first deployed in 1995 on autonomous vehicles.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T7",
        "question": "What are LLM limitations?",
        "answer": "LLM hallucinations are caused by next-token prediction objectives and knowledge cutoff limits. Language models do not possess an infinite memory capacity. Grounding answers with Retrieval-Augmented Generation helps reduce hallucination risk. Large language models have a knowledge cutoff in the year 2099.",
        "claims": [
            {
                "text": "LLM hallucinations are caused by next-token prediction objectives and knowledge cutoff limits.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "Language models do not possess an infinite memory capacity.",
                "gold_label": "SUPPORTED",  # Note: contains "not" -> triggers false alarm FP
                "gold_is_hallucination": False
            },
            {
                "text": "Grounding answers with Retrieval-Augmented Generation helps reduce hallucination risk.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "Large language models have a knowledge cutoff in the year 2099.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T8",
        "question": "Summarise BERT and the Transformer.",
        "answer": "The Transformer consists of an encoder stack and a decoder stack each with multi-head attention. BERT is designed to pre-train deep bidirectional representations. BERT can be fine-tuned with just one additional output layer for NLP tasks. Self-attention does not require recurrent connections to process tokens.",
        "claims": [
            {
                "text": "The Transformer consists of an encoder stack and a decoder stack each with multi-head attention.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "BERT is designed to pre-train deep bidirectional representations.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "BERT can be fine-tuned with just one additional output layer for NLP tasks.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "Self-attention does not require recurrent connections to process tokens.",
                "gold_label": "SUPPORTED",  # Note: contains "not" -> triggers false alarm FP
                "gold_is_hallucination": False
            }
        ]
    },
    {
        "id": "T9",
        "question": "Tell me about the GPT family.",
        "answer": "The Generative Pre-trained Transformer series was developed by OpenAI using autoregressive language modeling. GPT models use only the decoder part of the Transformer architecture. GPT-3 has 175 million parameters. GPT-4 was introduced in 1999 as an audio-only model.",
        "claims": [
            {
                "text": "The Generative Pre-trained Transformer series was developed by OpenAI using autoregressive language modeling.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "GPT models use only the decoder part of the Transformer architecture.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "GPT-3 has 175 million parameters.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            },
            {
                "text": "GPT-4 was introduced in 1999 as an audio-only model.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    },
    {
        "id": "T10",
        "question": "More AI history, please.",
        "answer": "Alan Turing proposed the Turing Test in 1950. IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997. AlexNet won the ImageNet competition in 2012. DeepMind's AlphaGo was created in 1850 by Ada Lovelace.",
        "claims": [
            {
                "text": "Alan Turing proposed the Turing Test in 1950.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "AlexNet won the ImageNet competition in 2012.",
                "gold_label": "SUPPORTED",
                "gold_is_hallucination": False
            },
            {
                "text": "DeepMind's AlphaGo was created in 1850 by Ada Lovelace.",
                "gold_label": "CONTRADICTED",
                "gold_is_hallucination": True
            }
        ]
    }
]

