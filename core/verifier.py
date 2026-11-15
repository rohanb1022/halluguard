import re
import time
from dataclasses import dataclass, asdict
from typing import List, Set, Dict, Any, Tuple, Optional
from .knowledge_base import KnowledgeBase


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", 
    "me", "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", 
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", 
    "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "without", 
    "cannot", "can't", "won't", "don't", "doesn't", "didn't", 
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't"
}


@dataclass
class ClaimResult:
    index: int
    text: str
    evidence: str
    source: str
    similarity: float
    coverage: float
    label: str  # "SUPPORTED", "CONTRADICTED", "UNSUPPORTED"
    reason: str
    mismatches: List[str]
    is_hallucination: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    flagged_claims: int
    hallucination_rate_pct: float
    response_time_ms: float
    claims: List[ClaimResult]
    annotated_html: str
    guarded_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "unsupported_claims": self.unsupported_claims,
            "flagged_claims": self.flagged_claims,
            "hallucination_rate_pct": self.hallucination_rate_pct,
            "response_time_ms": self.response_time_ms,
            "claims": [c.to_dict() for c in self.claims],
            "annotated_html": self.annotated_html,
            "guarded_answer": self.guarded_answer
        }


class HallucinationVerifier:
    """
    Lexical and Rule-based Claim-level Hallucination Detector (Doc Section 6.2).
    Extracts claims, retrieves matching evidence, validates entities, numbers/units,
    negation polarity, and content-word coverage to assign grounded verdicts.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Splits text into clean sentences, protecting common abbreviations."""
        if not text or not text.strip():
            return []
            
        clean_text = text.strip()
        # Protect common abbreviations like "et al.", "Dr.", "e.g.", "i.e."
        clean_text = re.sub(r'\bet al\.', 'et_al', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\be\.g\.', 'e_g', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\bi\.e\.', 'i_e', clean_text, flags=re.IGNORECASE)

        raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        sentences = []
        for s in raw_sentences:
            s_restored = s.replace('et_al', 'et al.').replace('e_g', 'e.g.').replace('i_e', 'i.e.').strip()
            if len(s_restored) > 0:
                sentences.append(s_restored)
        return sentences

    @staticmethod
    def extract_numbers(text: str) -> Set[str]:
        """
        Extracts numbers and quantities along with magnitude scale (e.g. 175 billion vs 175 million).
        """
        results = set()
        # Match number followed by optional scale unit (million, billion, trillion, k)
        pattern = r'\b(\d+(?:\.\d+)?)\s*(billion|million|trillion|thousand)?\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            num = match.group(1)
            unit = match.group(2)
            if unit:
                results.add(f"{num} {unit.lower()}")
            else:
                results.add(num)
        return results

    @staticmethod
    def extract_entities(text: str) -> Set[str]:
        """
        Extracts capitalized proper noun phrases (e.g., 'Alan Turing', 'Nobel Prize').
        """
        # Look for sequences of Capitalized Words (excluding first word of sentence if trivial)
        words = text.split()
        entities = set()
        current_entity = []

        for i, w in enumerate(words):
            cleaned = re.sub(r'[^\w-]', '', w)
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in STOP_WORDS:
                current_entity.append(cleaned)
            else:
                if current_entity:
                    entities.add(" ".join(current_entity))
                    current_entity = []
        if current_entity:
            entities.add(" ".join(current_entity))

        # Add single significant capitalized words as fallback
        for ent in list(entities):
            for part in ent.split():
                if len(part) > 3 and part.lower() not in STOP_WORDS:
                    entities.add(part)

        return entities

    @staticmethod
    def is_negated(text: str) -> bool:
        """Detects whether sentence contains explicit negation tokens."""
        tokens = re.findall(r'\b[\w\']+\b', text.lower())
        return any(t in NEGATION_WORDS for t in tokens)

    @staticmethod
    def content_words(text: str) -> Set[str]:
        """Extracts content words (filtered of punctuation and stop words)."""
        tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
        return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}

    def verify_claim(self, claim_text: str, index: int = 1) -> ClaimResult:
        """
        Verifies a single claim against the knowledge base following Doc Algorithm 6.2.
        """
        ev, src, sim = self.kb.retrieve(claim_text)
        
        ct = self.content_words(claim_text)
        et = self.content_words(ev)
        
        # Share of claim words found in evidence
        coverage = len(ct & et) / max(len(ct), 1)

        claim_nums = self.extract_numbers(claim_text)
        ev_nums = self.extract_numbers(ev)
        missing_nums = claim_nums - ev_nums

        claim_ents = self.extract_entities(claim_text)
        ev_ents = self.extract_entities(ev)
        ev_tokens_lower = set(re.findall(r'\b\w+\b', ev.lower()))
        
        # Missing entities that do not appear anywhere in evidence (case-insensitive)
        missing_ents = set()
        for ent in claim_ents:
            ent_lower = ent.lower()
            if ent not in ev_ents and not any(part in ev_tokens_lower for part in ent_lower.split()):
                missing_ents.add(ent)

        neg_claim = self.is_negated(claim_text)
        neg_ev = self.is_negated(ev)

        mismatches = []
        if missing_nums:
            mismatches.append(f"Number/Unit mismatch: {', '.join(sorted(missing_nums))}")
        if missing_ents and coverage < 0.70:
            mismatches.append(f"Entity mismatch: {', '.join(sorted(missing_ents))}")
        if neg_claim != neg_ev:
            mismatches.append("Negation polarity conflict")

        # Decision rules from PoC Document Section 6.2:
        if sim < 0.18 or coverage < 0.30:
            label = "UNSUPPORTED"
            reason = "No matching evidence in the knowledge base (possible fabrication / extrinsic hallucination)."
            is_hallucination = True
        elif neg_claim != neg_ev:
            label = "CONTRADICTED"
            reason = "Claim polarity directly contradicts the reference evidence."
            is_hallucination = True
        elif missing_nums:
            label = "CONTRADICTED"
            reason = f"Claim contains facts/numbers not in the matching evidence: {', '.join(sorted(missing_nums))}."
            is_hallucination = True
        elif missing_ents and coverage < 0.60:
            label = "CONTRADICTED"
            reason = f"Claim introduces ungrounded entities ({', '.join(sorted(missing_ents))}) contradicting source context."
            is_hallucination = True
        elif coverage >= 0.55:
            label = "SUPPORTED"
            reason = "Evidence covers the claim's key facts."
            is_hallucination = False
        else:
            label = "UNSUPPORTED"
            reason = "Evidence only partially matches claim; insufficient grounding to guarantee factual correctness."
            is_hallucination = True

        return ClaimResult(
            index=index,
            text=claim_text,
            evidence=ev,
            source=src,
            similarity=round(sim, 4),
            coverage=round(coverage, 3),
            label=label,
            reason=reason,
            mismatches=mismatches,
            is_hallucination=is_hallucination
        )

    def verify_answer(self, answer_text: str) -> VerificationReport:
        """
        Executes full pipeline: sentence splitting -> evidence retrieval -> claim verification ->
        guarded answer generation -> hallucination rate calculation.
        """
        start_time = time.perf_counter()
        
        sentences = self.split_sentences(answer_text)
        claims: List[ClaimResult] = []

        for idx, sentence in enumerate(sentences, start=1):
            claim_res = self.verify_claim(sentence, index=idx)
            claims.append(claim_res)

        total_claims = len(claims)
        supported = sum(1 for c in claims if c.label == "SUPPORTED")
        contradicted = sum(1 for c in claims if c.label == "CONTRADICTED")
        unsupported = sum(1 for c in claims if c.label == "UNSUPPORTED")
        flagged = contradicted + unsupported

        rate_pct = round((flagged / max(total_claims, 1)) * 100, 1)

        # Build Guarded Answer (only verified sentences with bracketed citations)
        guarded_parts = [f"{c.text} [{c.source}]" for c in claims if c.label == "SUPPORTED"]
        guarded_answer = " ".join(guarded_parts) if guarded_parts else "[No supported claims verified in this answer.]"

        # Build Annotated HTML (Green for supported, Red/Strikethrough for contradicted, Amber for unsupported)
        annotated_spans = []
        for c in claims:
            if c.label == "SUPPORTED":
                span = f'<span class="claim-badge badge-supported" data-id="{c.index}" title="SUPPORTED: {c.reason} [{c.source}]">{c.text}</span>'
            elif c.label == "CONTRADICTED":
                span = f'<span class="claim-badge badge-contradicted" data-id="{c.index}" title="CONTRADICTED: {c.reason}"><del>{c.text}</del></span>'
            else:
                span = f'<span class="claim-badge badge-unsupported" data-id="{c.index}" title="UNSUPPORTED: {c.reason}">{c.text}</span>'
            annotated_spans.append(span)

        annotated_html = " ".join(annotated_spans)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return VerificationReport(
            total_claims=total_claims,
            supported_claims=supported,
            contradicted_claims=contradicted,
            unsupported_claims=unsupported,
            flagged_claims=flagged,
            hallucination_rate_pct=rate_pct,
            response_time_ms=elapsed_ms,
            claims=claims,
            annotated_html=annotated_html,
            guarded_answer=guarded_answer
        )

