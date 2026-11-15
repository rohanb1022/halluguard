import os
import re
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    """
    Trusted Knowledge Base indexed with TF-IDF for cosine similarity retrieval.
    Stores factual sentences paired with their source file origin.
    """

    def __init__(self, kb_dir: Optional[str] = None):
        self.kb_dir = kb_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
        self.sentences: List[str] = []
        self.sources: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.file_contents: Dict[str, str] = {}
        self.load_kb()

    def _split_into_sentences(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sentences = []
        for line in lines:
            # Protect common abbreviations like "et al.", "e.g.", "i.e."
            clean_line = re.sub(r'\bet al\.', 'et_al', line, flags=re.IGNORECASE)
            clean_line = re.sub(r'\be\.g\.', 'e_g', clean_line, flags=re.IGNORECASE)
            clean_line = re.sub(r'\bi\.e\.', 'i_e', clean_line, flags=re.IGNORECASE)
            
            # Split by period, exclamation, question mark followed by space or end of string
            parts = re.split(r'(?<=[.!?])\s+', clean_line)
            for p in parts:
                cleaned = p.replace('et_al', 'et al.').replace('e_g', 'e.g.').replace('i_e', 'i.e.').strip()
                if len(cleaned) > 10:  # Ignore trivial fragments
                    sentences.append(cleaned)
        return sentences

    def load_kb(self):
        """Loads all .txt files from the knowledge base directory."""
        self.sentences = []
        self.sources = []
        self.file_contents = {}

        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir, exist_ok=True)

        txt_files = [f for f in os.listdir(self.kb_dir) if f.endswith(".txt")]
        txt_files.sort()

        for filename in txt_files:
            file_path = os.path.join(self.kb_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.file_contents[filename] = content
                    file_sentences = self._split_into_sentences(content)
                    for sent in file_sentences:
                        self.sentences.append(sent)
                        self.sources.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        self._fit_vectorizer()

    def load_custom_context(self, context_text: str, source_name: str = "custom_context.txt"):
        """Loads and indexes a custom context string provided by the user."""
        self.sentences = []
        self.sources = []
        self.file_contents = {source_name: context_text}
        
        file_sentences = self._split_into_sentences(context_text)
        for sent in file_sentences:
            self.sentences.append(sent)
            self.sources.append(source_name)
        
        self._fit_vectorizer()

    def _fit_vectorizer(self):
        """Fits or re-fits the TF-IDF vectorizer over all currently loaded sentences."""
        if not self.sentences:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.sentences)

    def retrieve(self, query: str, top_k: int = 1) -> Tuple[str, str, float]:
        """
        Retrieves the most semantically relevant evidence sentence from the KB.
        Returns: (evidence_sentence, source_filename, cosine_similarity_score)
        """
        if not self.sentences or self.vectorizer is None or self.tfidf_matrix is None:
            return "No reference knowledge available.", "unknown", 0.0

        clean_query = query.strip()
        if not clean_query:
            return self.sentences[0], self.sources[0], 0.0

        query_vec = self.vectorizer.transform([clean_query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        return self.sentences[best_idx], self.sources[best_idx], best_sim

    def retrieve_top_k(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-K candidate evidence sentences for detailed analysis."""
        if not self.sentences or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "sentence": self.sentences[idx],
                "source": self.sources[idx],
                "similarity": round(float(sims[idx]), 4)
            })
        return results

    def add_fact(self, sentence: str, source_file: str = "custom_facts.txt") -> bool:
        """Adds a single verified fact to the KB and updates the index."""
        cleaned = sentence.strip()
        if not cleaned:
            return False
        
        file_path = os.path.join(self.kb_dir, source_file)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{cleaned}")
        
        self.sentences.append(cleaned)
        self.sources.append(source_file)
        self._fit_vectorizer()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the indexed knowledge base."""
        return {
            "total_sentences": len(self.sentences),
            "files": list(self.file_contents.keys()),
            "file_counts": {
                filename: self.sources.count(filename)
                for filename in set(self.sources)
            }
        }
