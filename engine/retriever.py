# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Production RAG Retriever
  ────────────────────────────────────────────────────────────────────

  5-stage retrieval pipeline:

    Stage 1 — Semantic search          (FAISS cosine similarity)
    Stage 2 — Lexical search           (BM25 on Arabic tokens)
    Stage 3 — Hybrid fusion            (Reciprocal Rank Fusion)
    Stage 4 — Cross-encoder reranking  (pairwise query–passage scoring)
    Stage 5 — Metadata filtering       (category, answer quality)

  Backward compatible: the old FAISSRetriever API is preserved.
  New callers should use HybridRetriever directly.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import re
import pickle
import logging
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
#  Constants
# ═════════════════════════════════════════════════════════════════

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Cross-encoder for reranking (multilingual, small, fast)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Arabic stopwords to strip for BM25 tokenisation
_AR_STOPWORDS = frozenset([
    "في", "من", "على", "إلى", "عن", "مع", "هو", "هي", "هذا", "هذه",
    "ذلك", "تلك", "التي", "الذي", "ما", "لا", "أن", "إن", "كان", "يكون",
    "لم", "لن", "قد", "عند", "هل", "أو", "أي", "كل", "بعد", "قبل",
    "بين", "حتى", "ثم", "لكن", "ال", "و", "ب", "ل", "ف", "ك",
])


# ═════════════════════════════════════════════════════════════════
#  Arabic Tokeniser for BM25
# ═════════════════════════════════════════════════════════════════

def _tokenise_arabic(text: str) -> list[str]:
    """Lightweight Arabic tokeniser for BM25.

    - Strips diacritics
    - Normalises hamza/alef variants
    - Removes stopwords
    - Keeps tokens ≥ 2 chars
    """
    if not text:
        return []
    # Strip tashkeel (diacritics)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", text)
    # Normalise alef variants
    text = re.sub(r"[إأآا]", "ا", text)
    # Split on non-Arabic
    tokens = re.findall(r"[\u0600-\u06FF]+", text)
    return [t for t in tokens if len(t) >= 2 and t not in _AR_STOPWORDS]


# ═════════════════════════════════════════════════════════════════
#  Semantic Chunker
# ═════════════════════════════════════════════════════════════════

def semantic_chunk(text: str, max_chunk: int = 512, overlap: int = 64) -> list[str]:
    """Split a long answer into overlapping semantic chunks.

    Strategy:
      1. Split on sentence boundaries (Arabic period / newline)
      2. Greedily pack sentences into chunks ≤ max_chunk chars
      3. Overlap the last *overlap* characters for continuity

    Args:
        text:      Input text.
        max_chunk: Max characters per chunk.
        overlap:   Overlap in characters between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    if len(text) <= max_chunk:
        return [text]

    # Split on Arabic sentence boundaries
    sentences = re.split(r"(?<=[.،؟!\n])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chunk:
            current = f"{current} {sent}".strip() if current else sent
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from previous
            if overlap > 0 and current:
                tail = current[-overlap:]
                current = f"{tail} {sent}".strip()
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks


# ═════════════════════════════════════════════════════════════════
#  NLP Specialty Router (Zero-Shot Medical Domain Detection)
# ═════════════════════════════════════════════════════════════════

def detect_medical_specialty(query: str) -> str | None:
    """
    Detects the Arabic medical specialty from user query to enable strict domain filtering.
    E.g. map 'التهاب الكبد' -> 'الجهاز الهضمي والكبد'
    """
    query_clean = query.lower()
    
    specialties_map = {
        "الجهاز الهضمي والكبد": ["كبد", "معدة", "هضم", "قولون", "مرارة", "بنكرياس", "بلغم", "كبدية", "يرقان", "بواسير", "تقيؤ", "غثيان"],
        "أنف، أذن وحنجرة": ["أذن", "حنجرة", "أنف", "جيوب", "سمع", "شخير", "لوزتين", "رعاف", "صوت", "طنين"],
        "أمراض القلب والشرايين": ["قلب", "شريان", "ضغط", "وريد", "خفقان", "كولسترول", "جلطة", "صدرية", "نوبة"],
        "أمراض الجهاز التنفسي": ["تنفس", "رئة", "ربو", "سعال", "شعب", "صدر", "بلغم", "سل", "التهاب رئوي", "زكام"],
        "الأمراض الجلدية": ["جلد", "شعر", "صدفية", "حب شباب", "أظافر", "حكة", "حساسية", "ثعلبة", "بهاق", "اكزيما", "حروق", "جروح"],
        "أمراض العيون": ["عين", "رؤية", "بصر", "شبكية", "قرنية", "جلوكوما", "ماء أبيض", "جفاف", "عدسة"],
        "الأمراض النفسية": ["نفسي", "اكتئاب", "قلق", "توتر", "فصام", "وهم", "ذهان", "أرق", "نوم", "انفصام", "هلوسة", "تصلب"],
        "أمراض النساء والتوليد": ["حمل", "دورة", "رحم", "مبيض", "ولادة", "طمث", "إجهاض", "مهبل", "ثدي", "عقم"],
        "طب الأطفال": ["طفل", "رضيع", "نمو", "تطعيم", "حصبة", "جدري", "اطفال"],
        "جراحة العظام والمفاصل": ["عظم", "مفصل", "كسر", "غضروف", "روماتيزم", "ظهر", "عمود فقري", "ديسك", "ورك", "ركبة", "عضلة", "هشاشة", "صلب"],
        "المسالك البولية": ["بول", "كلى", "مثانة", "بروستاتا", "حصى", "خصية", "تناسلي", "تبول"],
        "أمراض الدم": ["دم", "أنيميا", "فقر دم", "لوكيميا", "نزيف", "تخثر", "هيموجلوبين"],
        "طب الأسنان": ["سن", "لثة", "ضرس", "عصب", "تسوس", "تقويم"],
        "أمراض الغدد الصماء": ["غدة", "سكري", "هرمون", "دراقية", "سمنة", "نحافة", "نخامية", "كظرية"],
        "الأمراض العصبية": ["عصب", "جلطة دماغية", "صرع", "شلل", "زهايمر", "صداع", "شقيقة", "دماغ", "رقبة", "نخاع"],
        "الأورام والمناعة": ["سرطان", "ورم", "خبيث", "حميد", "مناعة", "نقص"],
    }
    
    best_match = None
    max_hits = 0
    
    for spec, keywords in specialties_map.items():
        hits = sum(1 for kw in keywords if kw in query_clean)
        if hits > max_hits:
            max_hits = hits
            best_match = spec
            
    return best_match


# ═════════════════════════════════════════════════════════════════
#  Hybrid Retriever
# ═════════════════════════════════════════════════════════════════

class HybridRetriever:
    """Production-grade RAG retriever with 5-stage pipeline.

    Stages:
      1. FAISS dense retrieval          — semantic similarity
      2. BM25 sparse retrieval          — lexical keyword matching
      3. Reciprocal Rank Fusion (RRF)   — merges both ranked lists
      4. Cross-encoder reranking        — pairwise relevance scoring
      5. Metadata + quality filtering   — category priors, length checks

    The old FAISSRetriever interface (build_index, search, get_best_answer,
    encode_query, save, load) is fully preserved for backward compatibility.
    """

    INDEX_PATH = "models/faiss_index_camel.bin"
    DATA_PATH  = "models/faiss_index_camel_meta.pkl"

    # Stage weights
    RRF_K = 60                         # RRF smoothing constant
    FAISS_WEIGHT = 0.6                 # Proportion of dense score
    BM25_WEIGHT  = 0.4                 # Proportion of sparse score

    def __init__(self, *, enable_reranker: bool = True):
        # Dense
        self.index: Optional[faiss.Index] = None
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None

        # Data
        self.df = None
        self._questions: list[str] = []          # raw questions
        self._answers: list[str] = []            # raw answers
        self._tokenised_corpus: list[list[str]] = []

        # BM25
        self._bm25 = None

        # Reranker
        self._enable_reranker = enable_reranker
        self._reranker = None

    # ─────────────────────────────────────────────────────────────
    #  Model loading
    # ─────────────────────────────────────────────────────────────
    def load_model(self):
        """Load the bi-encoder embedding model."""
        if self.model is None:
            logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
            self.model = SentenceTransformer(EMBED_MODEL_NAME)
        return self.model

    def _load_reranker(self):
        """Lazy-load the cross-encoder reranker."""
        if self._reranker is None and self._enable_reranker:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL_NAME)
                self._reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=256)
            except Exception as e:
                logger.warning("Cross-encoder unavailable, skipping reranking: %s", e)
                self._enable_reranker = False
        return self._reranker

    def _build_bm25(self):
        """Build BM25 index from the tokenised corpus."""
        if not self._tokenised_corpus:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenised_corpus)
            logger.info("BM25 index built: %d documents", len(self._tokenised_corpus))
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 disabled. pip install rank-bm25")
            self._bm25 = None

    # ─────────────────────────────────────────────────────────────
    #  Build & persist
    # ─────────────────────────────────────────────────────────────
    def build_index(self, df, verbose: bool = True):
        """Build FAISS + BM25 indices from a KB DataFrame.

        The DataFrame must have columns: question, answer, category, intent.
        Long answers are semantically chunked before indexing.

        Returns:
            np.ndarray of embeddings (for classifier training).
        """
        self.df = df.reset_index(drop=True)
        self.load_model()

        # ── Semantic chunking ───────────────────────────────────
        # For each row, chunk the answer and create expanded entries
        # (The question is replicated for each chunk of its answer)
        expanded_questions = []
        expanded_answers = []
        expanded_categories = []
        expanded_intents = []
        expanded_source_idx = []       # Which original row each chunk came from

        for idx, row in self.df.iterrows():
            q = str(row["question"])
            a = str(row["answer"])
            cat = row.get("category", "عام")
            intent = row.get("intent", "")

            chunks = semantic_chunk(a, max_chunk=512, overlap=64)
            for chunk in chunks:
                expanded_questions.append(q)
                expanded_answers.append(chunk)
                expanded_categories.append(cat)
                expanded_intents.append(intent)
                expanded_source_idx.append(idx)

        self._questions = expanded_questions
        self._answers = expanded_answers

        if verbose:
            orig = len(df)
            chunked = len(expanded_questions)
            logger.info("Semantic chunking: %d → %d entries (%.1fx expansion)",
                        orig, chunked, chunked / max(orig, 1))
            print(f"\n  📐 تشفير {chunked} وحدة ({orig} سؤال أصلي)...")

        # ── Encode for FAISS ────────────────────────────────────
        self.embeddings = self.model.encode(
            expanded_questions,
            show_progress_bar=verbose,
            normalize_embeddings=True,
            batch_size=64,
        ).astype("float32")

        self.dimension = self.embeddings.shape[1]

        # Build FAISS (exact cosine via IP on normalised vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

        if verbose:
            print(f"    ✅ FAISS جاهز: {self.index.ntotal} متجه × {self.dimension} بُعد")

        # ── Build BM25 ──────────────────────────────────────────
        self._tokenised_corpus = [
            _tokenise_arabic(q) for q in expanded_questions
        ]
        self._build_bm25()

        if verbose and self._bm25:
            print(f"    ✅ BM25 جاهز: {len(self._tokenised_corpus)} مستند")

        # ── Store metadata for save/load ────────────────────────
        self._meta = {
            "categories": expanded_categories,
            "intents": expanded_intents,
            "source_idx": expanded_source_idx,
        }

        return self.embeddings

    def save(self, index_path: str = None, data_path: str = None):
        """Persist FAISS index + all metadata to disk."""
        index_path = index_path or self.INDEX_PATH
        data_path = data_path or self.DATA_PATH

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(data_path, "wb") as f:
            pickle.dump({
                "df": self.df,
                "embeddings": self.embeddings,
                "dimension": self.dimension,
                "questions": self._questions,
                "answers": self._answers,
                "tokenised_corpus": self._tokenised_corpus,
                "meta": self._meta,
            }, f)

        logger.info("Saved FAISS index → %s", index_path)
        logger.info("Saved metadata   → %s", data_path)
        print(f"    💾 تم حفظ الفهرس: {index_path}")

    def load(self, index_path: str = None, data_path: str = None) -> bool:
        """Load persisted FAISS + BM25 + metadata. Returns True on success."""
        index_path = index_path or self.INDEX_PATH
        data_path = data_path or self.DATA_PATH

        if not os.path.exists(index_path) or not os.path.exists(data_path):
            return False

        self.index = faiss.read_index(index_path)

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.df = data["df"]
        self.embeddings = data["embeddings"]
        self.dimension = data["dimension"]

        # Hybrid data (may be absent in old indices)
        self._questions = data.get("questions", [])
        self._answers = data.get("answers", [])
        self._tokenised_corpus = data.get("tokenised_corpus", [])
        self._meta = data.get("meta", {})

        # If loaded from old format, rebuild sparse data
        if not self._questions and self.df is not None:
            self._questions = self.df["question"].tolist()
            self._answers = self.df["answer"].tolist()
            self._tokenised_corpus = [_tokenise_arabic(q) for q in self._questions]
            self._meta = {
                "categories": self.df.get("category", "عام").tolist()
                              if "category" in self.df.columns else [],
                "intents": self.df.get("intent", "").tolist()
                            if "intent" in self.df.columns else [],
                "source_idx": list(range(len(self._questions))),
            }

        self._build_bm25()
        self.load_model()
        return True

    # ─────────────────────────────────────────────────────────────
    #  Stage 1 — Dense retrieval (FAISS)
    # ─────────────────────────────────────────────────────────────
    def _dense_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k candidates from FAISS."""
        if self.index is None or self.model is None:
            return []

        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            results.append({"idx": int(idx), "dense_score": float(score), "dense_rank": rank + 1})
        return results

    # ─────────────────────────────────────────────────────────────
    #  Stage 2 — Sparse retrieval (BM25)
    # ─────────────────────────────────────────────────────────────
    def _sparse_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k candidates from BM25."""
        if self._bm25 is None:
            return []

        q_tokens = _tokenise_arabic(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            results.append({"idx": int(idx), "bm25_score": float(scores[idx]), "bm25_rank": rank + 1})
        return results

    # ─────────────────────────────────────────────────────────────
    #  Stage 3 — Reciprocal Rank Fusion
    # ─────────────────────────────────────────────────────────────
    def _rrf_fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """Merge dense + sparse with RRF scoring.

        RRF score = w₁ / (k + rank_dense) + w₂ / (k + rank_sparse)
        """
        k = self.RRF_K
        fused: dict[int, dict] = {}

        for r in dense_results:
            idx = r["idx"]
            fused[idx] = {
                "idx": idx,
                "dense_score": r["dense_score"],
                "dense_rank": r["dense_rank"],
                "bm25_score": 0.0,
                "bm25_rank": 999,
                "rrf_score": (self.FAISS_WEIGHT / (k + r["dense_rank"])) * 100,
            }

        for r in sparse_results:
            idx = r["idx"]
            if idx in fused:
                fused[idx]["bm25_score"] = r["bm25_score"]
                fused[idx]["bm25_rank"] = r["bm25_rank"]
                fused[idx]["rrf_score"] += (self.BM25_WEIGHT / (k + r["bm25_rank"])) * 100
            else:
                fused[idx] = {
                    "idx": idx,
                    "dense_score": 0.0,
                    "dense_rank": 999,
                    "bm25_score": r["bm25_score"],
                    "bm25_rank": r["bm25_rank"],
                    "rrf_score": (self.BM25_WEIGHT / (k + r["bm25_rank"])) * 100,
                }

        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked

    # ─────────────────────────────────────────────────────────────
    #  Stage 4 — Cross-encoder reranking
    # ─────────────────────────────────────────────────────────────
    def _rerank(self, query: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
        """Score (query, passage) pairs with a cross-encoder.

        Falls back to RRF order if the reranker is unavailable.
        """
        reranker = self._load_reranker()
        if reranker is None or not candidates:
            return candidates[:top_k]

        # Build pairs: (query, question + answer)
        pairs = []
        for c in candidates[:top_k * 2]:   # feed 2x to reranker
            idx = c["idx"]
            q_text = self._questions[idx] if idx < len(self._questions) else ""
            a_text = self._answers[idx] if idx < len(self._answers) else ""
            passage = f"{q_text}\n{a_text}"
            pairs.append((query, passage))

        try:
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            for i, c in enumerate(candidates[:len(ce_scores)]):
                c["ce_score"] = float(ce_scores[i])
            # Sort by cross-encoder score
            reranked = sorted(
                candidates[:len(ce_scores)],
                key=lambda x: x.get("ce_score", -999),
                reverse=True,
            )
            return reranked[:top_k]
        except Exception as e:
            logger.warning("Reranking failed, using RRF order: %s", e)
            return candidates[:top_k]

    # ─────────────────────────────────────────────────────────────
    #  Stage 5 — Metadata filtering + quality scoring
    # ─────────────────────────────────────────────────────────────
    def _filter_and_score(
        self,
        candidates: list[dict],
        *,
        category_filter: str | None = None,
        min_answer_len: int = 30,
        threshold: float = 0.20,
    ) -> list[dict]:
        """Apply metadata filters and quality heuristics.

        Filters:
          - category_filter: only keep results matching this category
          - min_answer_len:  drop very short answers
          - threshold:       minimum RRF or CE score

        Enriches each candidate with question, answer, category, intent.
        """
        categories = self._meta.get("categories", [])
        intents    = self._meta.get("intents", [])

        enriched = []
        for c in candidates:
            idx = c["idx"]

            # Bounds check
            if idx >= len(self._questions):
                continue

            answer = self._answers[idx]
            category = categories[idx] if idx < len(categories) else "عام"
            intent = intents[idx] if idx < len(intents) else ""

            # Length filter
            if len(answer) < min_answer_len:
                continue

            # NLP Medical Specialty Filtering (Avoid Topic Mismatch)
            # If the user queried for Liver (Gastroenterology), penalize or reject ENT results massively!
            if category_filter and category not in ["عام", "غير محدد"]:
                # Token overlap overlap mechanism (since exact strings like 'أمراض الجهاز الهضمي' vs 'الجهاز الهضمي' can differ)
                overlap = any(term in category for term in _tokenise_arabic(category_filter))
                if not overlap and category_filter not in category:
                    # Massive penalty for domain mismatch
                    if c.get("dense_score", 0) < 0.65: # Allow really strong universal Semantic embeddings, otherwise Kill it!
                        continue
                    # Severely penalize scores if it sneaks through the threshold
                    c["ce_score"] = float(c.get("ce_score", c.get("rrf_score", 0))) * 0.1
                    c["rrf_score"] = float(c.get("rrf_score", 0)) * 0.1
                    c["dense_score"] = float(c.get("dense_score", 0)) * 0.5
                
            # Hardware filter: Restrict absolute garbage semantic matches from passing through RRF
            if not self._enable_reranker and c.get("dense_score", 0.0) < 0.45:
                continue

            # Score threshold (use CE score if available, else RRF)
            score = c.get("ce_score", c.get("rrf_score", 0))
            if score < threshold:
                continue

            c.update({
                "question": self._questions[idx],
                "answer": answer,
                "category": category,
                "intent": intent,
                "final_score": score,
            })
            enriched.append(c)

        return enriched

    # ═════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.35,
        category: str | None = None,
    ) -> list[dict]:
        """Full hybrid search pipeline with NLP Auto-Routing.

        Args:
            query:     Arabic text query.
            top_k:     Number of final results.
            threshold: Minimum score.
            category:  Optional category filter.

        Returns:
            List of dicts with: question, answer, category, intent,
            final_score, dense_score, bm25_score, ce_score, rank.
        """
        # 1. NLP Specialty routing override (if none provided by the flow, fallback to smart detection
        detected_category = category if category else detect_medical_specialty(query)

        # Stage 1 + 2: parallel retrieval
        dense   = self._dense_search(query, top_k=20)
        sparse  = self._sparse_search(query, top_k=20)

        # Stage 3: fusion
        fused = self._rrf_fuse(dense, sparse)

        # Stage 4: cross-encoder reranking (top 12 → top_k)
        reranked = self._rerank(query, fused, top_k=top_k * 2)

        # Stage 5: metadata filtering + quality
        results = self._filter_and_score(
            reranked,
            category_filter=detected_category,
            threshold=threshold,
        )

        # Assign final ranks
        for i, r in enumerate(results[:top_k]):
            r["rank"] = i + 1
            r["score"] = r["final_score"]    # backward compat

        return results[:top_k]

    def get_best_answer(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.20,
    ):
        """Return the single best answer (backward-compatible).

        Returns:
            (answer, score, category, intent) or (None, 0.0, None, None)
        """
        results = self.search(query, top_k=top_k, threshold=threshold)
        if not results:
            return None, 0.0, None, None

        best = results[0]
        return best["answer"], best["final_score"], best["category"], best["intent"]

    def get_top_contexts(
        self,
        query: str,
        *,
        top_k: int = 3,
        category: str | None = None,
    ) -> list[dict]:
        """Return top-K high-quality contexts for LLM enrichment.

        This is the **recommended API** for the RAG agent.

        Returns:
            List of dicts, each containing: question, answer, category,
            intent, final_score, and all intermediate scores.
        """
        return self.search(query, top_k=top_k, threshold=0.15, category=category)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query for use with the intent classifier."""
        self.load_model()
        return self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")[0]

# Preserve backwards compatibility
FAISSRetriever = HybridRetriever
