"""
═══════════════════════════════════════════════════════════════════════
  الخطوة 3: الباحث (المسترجع)
  Step 3: The Retriever (The Searcher)

  Ultra-fast vector retrieval using FAISS (Facebook AI Similarity
  Search). Pre-indexes all question embeddings for sub-millisecond
  nearest-neighbour lookup.
═══════════════════════════════════════════════════════════════════════
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── Embedding model (multilingual, small, fast) ──────────────────
EMBED_MODEL_NAME = "aubmindlab/bert-base-arabertv2"


class FAISSRetriever:
    """
    Fast semantic retriever backed by FAISS IndexFlatIP (inner product).
    
    Workflow:
      1. Encode all KB questions into dense vectors
      2. Build a FAISS index (normalized → cosine similarity via IP)
      3. At query time: encode query → FAISS search → return top-K results
    """

    INDEX_PATH = "models/faiss_index_camel.bin"
    DATA_PATH  = "models/faiss_index_camel_meta.pkl"

    def __init__(self):
        self.index = None
        self.model = None
        self.df = None
        self.embeddings = None
        self.dimension = None

    def load_model(self):
        """Load the sentence-transformer embedding model."""
        if self.model is None:
            print(f"  🔤 جاري تحميل نموذج التضمين ({EMBED_MODEL_NAME})...")
            self.model = SentenceTransformer(EMBED_MODEL_NAME)
        return self.model

    def build_index(self, df, verbose: bool = True):
        """
        Build FAISS index from a knowledge base DataFrame.
        
        Args:
            df: DataFrame with 'question', 'answer', 'category', 'intent' columns
        """
        self.df = df.reset_index(drop=True)
        self.load_model()

        if verbose:
            print(f"\n  📐 تشفير {len(df)} سؤال إلى متجهات...")

        # Encode all questions
        questions = df['question'].tolist()
        self.embeddings = self.model.encode(
            questions,
            show_progress_bar=verbose,
            normalize_embeddings=True,    # L2-normalise → IP = cosine sim
            batch_size=64,
        ).astype('float32')

        self.dimension = self.embeddings.shape[1]

        # Build FAISS index  (IndexFlatIP for exact cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)

        if verbose:
            print(f"    ✅ فهرس FAISS جاهز: {self.index.ntotal} متجه بأبعاد {self.dimension}")

        return self.embeddings

    def search(self, query: str, top_k: int = 5, threshold: float = 0.35):
        """
        Search the FAISS index for the most similar questions.
        
        Args:
            query: user's question in Arabic
            top_k: number of results to return
            threshold: minimum similarity score
            
        Returns:
            List of dicts with keys: question, answer, category, intent, score, rank
        """
        if self.index is None or self.model is None:
            return []

        # Encode query
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype('float32')

        # FAISS search
        scores, indices = self.index.search(q_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0 or score < threshold:
                continue
            row = self.df.iloc[idx]
            results.append({
                "question": row['question'],
                "answer": row['answer'],
                "category": row.get('category', 'عام'),
                "intent": row.get('intent', ''),
                "score": float(score),
                "rank": rank + 1,
            })

        return results

    def get_best_answer(self, query: str, top_k: int = 5, threshold: float = 0.35):
        """
        Get the single best answer with smart re-ranking.
        
        Re-ranking considers:
          - Semantic similarity score
          - Answer length quality (penalise very short answers)
          - Keyword overlap bonus
          
        Returns:
            (answer, score, category, intent) or (None, 0.0, None, None)
        """
        results = self.search(query, top_k=top_k, threshold=threshold)

        if not results:
            return None, 0.0, None, None

        # Re-rank with heuristics
        query_words = set(query.split())
        best = None
        best_score = -1

        for r in results:
            ans = r["answer"]
            ans_words = set(ans.split())

            # Base score from FAISS
            score = r["score"]

            # Keyword overlap bonus (up to +0.1)
            if query_words:
                overlap = len(query_words & ans_words) / (len(query_words) + 1)
                score += 0.05 * overlap

            # Penalise very short answers
            if len(ans) < 50:
                score -= 0.12
            elif len(ans) > 100:
                score += 0.03  # Slight bonus for detailed answers

            if score > best_score:
                best_score = score
                best = r

        if best is None:
            return None, 0.0, None, None

        return best["answer"], best["score"], best["category"], best["intent"]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query for use with the classifier."""
        self.load_model()
        return self.model.encode(
            [query], normalize_embeddings=True
        ).astype('float32')[0]

    def save(self, index_path: str = None, data_path: str = None):
        """Save FAISS index and associated data."""
        index_path = index_path or self.INDEX_PATH
        data_path = data_path or self.DATA_PATH

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save DataFrame and metadata
        with open(data_path, "wb") as f:
            pickle.dump({
                "df": self.df,
                "embeddings": self.embeddings,
                "dimension": self.dimension,
            }, f)

        print(f"    💾 تم حفظ فهرس FAISS: {index_path}")
        print(f"    💾 تم حفظ البيانات: {data_path}")

    def load(self, index_path: str = None, data_path: str = None) -> bool:
        """Load previously saved FAISS index and data. Returns True on success."""
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

        self.load_model()
        return True
