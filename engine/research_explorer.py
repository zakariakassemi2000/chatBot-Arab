# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI · Research Explorer Engine
  ────────────────────────────────────────────────────────────────────
  Multi-source content search engine that finds and ranks:
    • Books     (Google Books API — free, no key required)
    • Articles  (Wikipedia API — free)
    • Videos    (YouTube Data API v3 — requires key)

  Semantic ranking powered by sentence-transformers (local, free).
  Composite score: 60% semantic + 25% popularity + 15% recency.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import hashlib
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("shifa.research_explorer")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")

RANKING_WEIGHTS = {
    "semantic": 0.60,
    "popularity": 0.25,
    "recency": 0.15,
}

CACHE_DIR = Path(__file__).parent.parent / "data" / "research_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 3600  # 1 hour


# ─────────────────────────────────────────────────────────────
# SEMANTIC RANKER (sentence-transformers)
# ─────────────────────────────────────────────────────────────

class SemanticRanker:
    """Ranks content items by cosine similarity using sentence-transformers."""

    _model = None  # class-level singleton

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                logger.info("SemanticRanker model loaded: paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                logger.warning("sentence-transformers unavailable, using fallback: %s", e)
        return cls._model

    def encode(self, text: str) -> Optional[np.ndarray]:
        model = self._load_model()
        if model is not None:
            try:
                return model.encode(text, normalize_embeddings=True)
            except Exception as e:
                logger.error("Encoding error: %s", e)
        return self._fallback_encode(text)

    @staticmethod
    def _fallback_encode(text: str) -> np.ndarray:
        """Ultra-simple bag-of-chars fallback when no model is available."""
        vec = np.zeros(128, dtype=np.float32)
        for ch in text.lower():
            vec[ord(ch) % 128] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0

    # ── Popularity ──────────────────────────────────────────
    @staticmethod
    def popularity_score(item: Dict[str, Any]) -> float:
        kind = item.get("type", "")
        if kind == "book":
            avg = item.get("average_rating", 0) or 0
            cnt = item.get("ratings_count", 0) or 0
            return min(1.0, (avg / 5.0) * 0.5 + min(cnt / 500, 1.0) * 0.5)
        if kind == "video":
            views = int(item.get("view_count", 0) or 0)
            return min(1.0, math.log10(max(views, 1)) / 7.0)
        if kind == "article":
            refs = item.get("references", 0) or 0
            return min(1.0, refs / 50.0)
        return 0.5

    # ── Recency ─────────────────────────────────────────────
    @staticmethod
    def recency_score(item: Dict[str, Any]) -> float:
        date_str = item.get("published_date") or item.get("published_at") or ""
        if not date_str:
            return 0.5
        try:
            # Try multiple date formats
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m", "%Y"):
                try:
                    dt = datetime.strptime(date_str[:len(fmt.replace("%", "x"))], fmt)
                    break
                except ValueError:
                    continue
            else:
                # Handle year-only strings
                year_match = re.match(r"(\d{4})", date_str)
                if year_match:
                    dt = datetime(int(year_match.group(1)), 1, 1)
                else:
                    return 0.5

            age_years = (datetime.now() - dt).days / 365.25
            if age_years <= 1:
                return 1.0
            elif age_years <= 3:
                return 0.8
            elif age_years <= 5:
                return 0.6
            elif age_years <= 10:
                return 0.4
            return 0.2
        except Exception:
            return 0.5

    # ── Combined ranking ────────────────────────────────────
    def rank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank items by composite score (semantic + popularity + recency)."""
        if not items:
            return []

        query_emb = self.encode(query)
        w = RANKING_WEIGHTS

        for item in items:
            text = " ".join(filter(None, [
                item.get("title", ""),
                item.get("description", "") or item.get("summary", ""),
                " ".join(item.get("authors", [])) if isinstance(item.get("authors"), list) else "",
            ]))
            item_emb = self.encode(text)
            sem = self.cosine_similarity(query_emb, item_emb)
            pop = self.popularity_score(item)
            rec = self.recency_score(item)

            item["semantic_score"] = round(sem, 4)
            item["popularity_score"] = round(pop, 4)
            item["recency_score"] = round(rec, 4)
            item["relevance_score"] = round(
                w["semantic"] * sem + w["popularity"] * pop + w["recency"] * rec, 4
            )

        return sorted(items, key=lambda x: x.get("relevance_score", 0), reverse=True)


# ─────────────────────────────────────────────────────────────
# CONTENT SEARCH ENGINE
# ─────────────────────────────────────────────────────────────

class ContentSearchEngine:
    """Unified search across Google Books, Wikipedia, and YouTube."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "SHIFA-AI/2.0"})
        self._ranker = SemanticRanker()

    # ════════════════════════════════════════════════════════
    #  BOOKS — Google Books API
    # ════════════════════════════════════════════════════════

    def search_books(self, query: str, max_results: int = 5, lang: str = "ar") -> List[Dict[str, Any]]:
        """Search Google Books API, with OpenLibrary fallback."""
        books = self._search_books_google(query, max_results, lang)
        if not books:
            logger.info("Google Books returned 0 results, trying OpenLibrary fallback...")
            books = self._search_books_openlibrary(query, max_results)
        return books

    def _search_books_google(self, query: str, max_results: int = 5, lang: str = "ar") -> List[Dict[str, Any]]:
        """Search Google Books API. No API key required for basic usage."""
        try:
            params: Dict[str, Any] = {
                "q": query,
                "maxResults": min(max_results, 10),
                "printType": "books",
                "orderBy": "relevance",
            }
            if lang:
                params["langRestrict"] = lang
            if GOOGLE_BOOKS_API_KEY:
                params["key"] = GOOGLE_BOOKS_API_KEY

            resp = self._session.get(
                "https://www.googleapis.com/books/v1/volumes",
                params=params,
                timeout=8,
            )
            if resp.status_code == 429:
                logger.warning("Google Books quota exceeded (HTTP 429), using fallback")
                return []
            if resp.status_code != 200:
                logger.warning("Google Books HTTP %d", resp.status_code)
                if lang:
                    return self._search_books_google(query, max_results, lang="")
                return []

            data = resp.json()
            books: List[Dict[str, Any]] = []
            for item in data.get("items", []):
                vi = item.get("volumeInfo", {})
                si = item.get("searchInfo", {})
                
                # Fetch description with hierarchy: volumeInfo.description -> searchInfo.textSnippet
                raw_desc = vi.get("description") or si.get("textSnippet") or ""
                
                books.append({
                    "type": "book",
                    "id": item.get("id", ""),
                    "title": vi.get("title", "بدون عنوان"),
                    "subtitle": vi.get("subtitle", ""),
                    "authors": vi.get("authors", ["مؤلف غير معروف"]),
                    "description": raw_desc[:500],
                    "publisher": vi.get("publisher", ""),
                    "published_date": vi.get("publishedDate", ""),
                    "page_count": vi.get("pageCount", 0),
                    "categories": vi.get("categories", []),
                    "language": vi.get("language", ""),
                    "thumbnail": vi.get("imageLinks", {}).get("thumbnail", ""),
                    "preview_link": vi.get("previewLink", ""),
                    "info_link": vi.get("infoLink", ""),
                    "average_rating": vi.get("averageRating", 0),
                    "ratings_count": vi.get("ratingsCount", 0),
                })
            return books

        except Exception as e:
            logger.error("Google Books search error: %s", e)
            return []

    def _search_books_openlibrary(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Fallback: search Open Library (completely free, no API key, no quota)."""
        try:
            resp = self._session.get(
                "https://openlibrary.org/search.json",
                params={
                    "q": query,
                    "limit": min(max_results, 10),
                    "fields": "key,title,subtitle,author_name,first_publish_year,publisher,number_of_pages_median,subject,cover_i,ratings_average,ratings_count,edition_count,language",
                },
                timeout=12,
            )
            if resp.status_code != 200:
                logger.warning("OpenLibrary HTTP %d", resp.status_code)
                return []

            data = resp.json()
            books: List[Dict[str, Any]] = []
            for doc in data.get("docs", [])[:max_results]:
                cover_id = doc.get("cover_i")
                thumbnail = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""
                ol_key = doc.get("key", "")
                info_link = f"https://openlibrary.org{ol_key}" if ol_key else ""

                subjects = doc.get("subject", [])
                categories = subjects[:3] if isinstance(subjects, list) else []

                books.append({
                    "type": "book",
                    "id": ol_key,
                    "title": doc.get("title", "بدون عنوان"),
                    "subtitle": doc.get("subtitle", ""),
                    "authors": doc.get("author_name", ["مؤلف غير معروف"]),
                    "description": ", ".join(categories) if categories else "",
                    "publisher": (doc.get("publisher", [""])[0] if isinstance(doc.get("publisher"), list) else ""),
                    "published_date": str(doc.get("first_publish_year", "")),
                    "page_count": doc.get("number_of_pages_median", 0) or 0,
                    "categories": categories,
                    "language": (doc.get("language", [""])[0] if isinstance(doc.get("language"), list) else ""),
                    "thumbnail": thumbnail,
                    "preview_link": info_link,
                    "info_link": info_link,
                    "average_rating": doc.get("ratings_average", 0) or 0,
                    "ratings_count": doc.get("ratings_count", 0) or 0,
                })
            logger.info("OpenLibrary returned %d books", len(books))
            return books

        except Exception as e:
            logger.error("OpenLibrary search error: %s", e)
            return []

    # ════════════════════════════════════════════════════════
    #  ARTICLES — Wikipedia API
    # ════════════════════════════════════════════════════════

    def search_articles(self, query: str, max_results: int = 3, lang: str = "ar") -> List[Dict[str, Any]]:
        """Search Wikipedia via its REST API (no package dependency)."""
        articles: List[Dict[str, Any]] = []

        for wiki_lang in [lang, "fr", "en"]:
            if len(articles) >= max_results:
                break
            try:
                # Step 1: Search for matching titles
                search_resp = self._session.get(
                    f"https://{wiki_lang}.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": max_results,
                        "format": "json",
                        "utf8": 1,
                    },
                    timeout=10,
                )
                if search_resp.status_code != 200:
                    continue

                search_data = search_resp.json()
                results = search_data.get("query", {}).get("search", [])

                for r in results:
                    if len(articles) >= max_results:
                        break
                    page_id = r.get("pageid")
                    title = r.get("title", "")

                    # Step 2: Get page summary
                    summary_resp = self._session.get(
                        f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                        timeout=10,
                    )
                    if summary_resp.status_code != 200:
                        continue

                    summary_data = summary_resp.json()
                    extract = summary_data.get("extract", "")[:600]

                    articles.append({
                        "type": "article",
                        "id": str(page_id),
                        "title": summary_data.get("title", title),
                        "summary": extract + ("..." if len(summary_data.get("extract", "")) > 600 else ""),
                        "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        "thumbnail": summary_data.get("thumbnail", {}).get("source", ""),
                        "source": f"Wikipedia ({wiki_lang.upper()})",
                        "lang": wiki_lang,
                        "references": r.get("wordcount", 0) // 50,  # rough proxy
                        "description": summary_data.get("description", ""),
                    })

            except Exception as e:
                logger.error("Wikipedia search error (%s): %s", wiki_lang, e)

        return articles[:max_results]

    # ════════════════════════════════════════════════════════
    #  VIDEOS — YouTube Data API v3
    # ════════════════════════════════════════════════════════

    def search_videos(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search YouTube. Falls back to Invidious public API if no key."""
        if YOUTUBE_API_KEY:
            return self._search_youtube_official(query, max_results)
        return self._search_youtube_fallback(query, max_results)

    def _search_youtube_official(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        try:
            params = {
                "part": "snippet",
                "q": query,
                "maxResults": max_results,
                "type": "video",
                "relevanceLanguage": "ar",
                "key": YOUTUBE_API_KEY,
            }
            resp = self._session.get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params,
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("YouTube API HTTP %d", resp.status_code)
                return self._search_youtube_fallback(query, max_results)

            data = resp.json()
            video_ids = [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]
            stats = self._get_youtube_stats(video_ids) if video_ids else {}

            videos: List[Dict[str, Any]] = []
            for item in data.get("items", []):
                vid = item.get("id", {}).get("videoId", "")
                if not vid:
                    continue
                snippet = item.get("snippet", {})
                st = stats.get(vid, {})
                videos.append({
                    "type": "video",
                    "id": vid,
                    "title": snippet.get("title", ""),
                    "description": (snippet.get("description") or "")[:300],
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "view_count": int(st.get("viewCount", 0)),
                    "like_count": int(st.get("likeCount", 0)),
                })
            return videos
        except Exception as e:
            logger.error("YouTube official API error: %s", e)
            return self._search_youtube_fallback(query, max_results)

    def _get_youtube_stats(self, video_ids: List[str]) -> Dict[str, Dict]:
        try:
            resp = self._session.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "statistics",
                    "id": ",".join(video_ids),
                    "key": YOUTUBE_API_KEY,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return {it["id"]: it.get("statistics", {}) for it in resp.json().get("items", [])}
        except Exception as e:
            logger.error("YouTube stats error: %s", e)
        return {}

    def _search_youtube_fallback(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback: try multiple Piped/Invidious instances (public, no key needed)."""
        piped_instances = [
            "https://pipedapi.kavin.rocks",
            "https://pipedapi.adminforge.de",
            "https://api.piped.yt",
            "https://pipedapi.in.projectsegfau.lt",
        ]

        for base_url in piped_instances:
            try:
                resp = self._session.get(
                    f"{base_url}/search",
                    params={"q": query, "filter": "videos"},
                    timeout=8,
                )
                if resp.status_code != 200:
                    logger.warning("Piped API (%s) HTTP %d", base_url, resp.status_code)
                    continue

                data = resp.json()

                # Handle both response formats: list or dict with "items" key
                if isinstance(data, list):
                    items_list = data
                elif isinstance(data, dict):
                    items_list = data.get("items", data.get("results", []))
                else:
                    continue

                if not items_list:
                    continue

                videos: List[Dict[str, Any]] = []
                for item in items_list[:max_results]:
                    vid_url = item.get("url", "")
                    # Extract video ID from various URL formats
                    if "v=" in vid_url:
                        vid_id = vid_url.split("v=")[-1].split("&")[0]
                    elif "/watch?v=" in vid_url:
                        vid_id = vid_url.replace("/watch?v=", "").split("&")[0]
                    elif vid_url.startswith("/watch"):
                        vid_id = vid_url.split("v=")[-1].split("&")[0] if "v=" in vid_url else vid_url.lstrip("/")
                    else:
                        vid_id = vid_url.lstrip("/")

                    if not vid_id or vid_id == "#":
                        continue

                    videos.append({
                        "type": "video",
                        "id": vid_id,
                        "title": item.get("title", ""),
                        "description": (item.get("shortDescription") or item.get("description") or "")[:300],
                        "channel_title": item.get("uploaderName", item.get("uploader", "")),
                        "published_at": item.get("uploadedDate", item.get("uploaded", "")),
                        "thumbnail": item.get("thumbnail", ""),
                        "url": f"https://www.youtube.com/watch?v={vid_id}",
                        "view_count": item.get("views", 0) or 0,
                        "like_count": 0,
                    })

                if videos:
                    logger.info("Piped (%s) returned %d videos", base_url, len(videos))
                    return videos

            except requests.exceptions.Timeout:
                logger.warning("Piped API (%s) timed out", base_url)
                continue
            except Exception as e:
                logger.error("Piped API (%s) error: %s", base_url, e)
                continue

        # Final fallback: try Invidious API
        return self._search_youtube_invidious(query, max_results)

    def _search_youtube_invidious(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Final fallback: Invidious public instances."""
        invidious_instances = [
            "https://vid.puffyan.us",
            "https://invidious.fdn.fr",
            "https://inv.nadeko.net",
            "https://invidious.privacyredirect.com",
        ]

        for base_url in invidious_instances:
            try:
                resp = self._session.get(
                    f"{base_url}/api/v1/search",
                    params={"q": query, "type": "video", "sort_by": "relevance"},
                    timeout=8,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                if not isinstance(data, list):
                    continue

                videos: List[Dict[str, Any]] = []
                for item in data[:max_results]:
                    if item.get("type") != "video":
                        continue
                    vid_id = item.get("videoId", "")
                    if not vid_id:
                        continue

                    # Get best thumbnail
                    thumbs = item.get("videoThumbnails", [])
                    thumb_url = ""
                    for t in thumbs:
                        if t.get("quality") in ("high", "medium", "default"):
                            thumb_url = t.get("url", "")
                            break
                    if not thumb_url and thumbs:
                        thumb_url = thumbs[0].get("url", "")

                    videos.append({
                        "type": "video",
                        "id": vid_id,
                        "title": item.get("title", ""),
                        "description": (item.get("description") or "")[:300],
                        "channel_title": item.get("author", ""),
                        "published_at": item.get("publishedText", ""),
                        "thumbnail": thumb_url,
                        "url": f"https://www.youtube.com/watch?v={vid_id}",
                        "view_count": item.get("viewCount", 0) or 0,
                        "like_count": 0,
                    })

                if videos:
                    logger.info("Invidious (%s) returned %d videos", base_url, len(videos))
                    return videos

            except requests.exceptions.Timeout:
                logger.warning("Invidious (%s) timed out", base_url)
                continue
            except Exception as e:
                logger.error("Invidious (%s) error: %s", base_url, e)
                continue

        logger.warning("All YouTube fallback instances failed")
        return []

    # ════════════════════════════════════════════════════════
    #  TOPIC DETECTION (via Groq LLM)
    # ════════════════════════════════════════════════════════

    def detect_topic(self, query: str) -> Dict[str, str]:
        """Detect the medical topic and optimize search terms using Groq."""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return {"topic": query, "category": "عام", "search_query_ar": query, "search_query_en": query}

            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{
                    "role": "system",
                    "content": "أنت محلل طبي. أجب فقط بصيغة JSON بدون أي نص إضافي."
                }, {
                    "role": "user",
                    "content": (
                        f"حلل هذا الاستفسار وأعط: 1) الموضوع الرئيسي 2) التصنيف (صحة/علوم/تغذية/نفسي/عام) "
                        f"3) كلمات البحث المحسنة بالعربية 4) كلمات البحث بالإنجليزية.\n"
                        f"الاستفسار: {query}\n"
                        f'أجب بصيغة JSON: {{"topic":"...","category":"...","search_query_ar":"...","search_query_en":"..."}}'
                    )
                }],
                temperature=0.1,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Topic detection error (using query as-is): %s", e)

        return {"topic": query, "category": "عام", "search_query_ar": query, "search_query_en": query}

    # ════════════════════════════════════════════════════════
    #  MAIN SEARCH PIPELINE
    # ════════════════════════════════════════════════════════

    def search_all(
        self,
        query: str,
        *,
        max_books: int = 5,
        max_articles: int = 3,
        max_videos: int = 3,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline: detect topic → search all sources → rank → return.

        Returns:
            {
                "query": str,
                "topic": {...},
                "books": [...],
                "articles": [...],
                "videos": [...],
                "timestamp": str,
            }
        """
        # ── Cache check ──
        cache_key = hashlib.md5(f"{query}_{max_books}_{max_articles}_{max_videos}".encode()).hexdigest()
        if use_cache:
            cached = self._read_cache(cache_key)
            if cached:
                return cached

        # ── Topic detection ──
        topic = self.detect_topic(query)
        search_ar = topic.get("search_query_ar", query)
        search_en = topic.get("search_query_en", query)

        # ── Parallel-ish search (sequential for simplicity) ──
        books = self.search_books(search_ar, max_books) or self.search_books(search_en, max_books, lang="en")
        articles = self.search_articles(search_ar, max_articles)
        videos = self.search_videos(search_en, max_videos)

        # ── Semantic ranking ──
        ranked_books = self._ranker.rank(query, books)[:max_books]
        ranked_articles = self._ranker.rank(query, articles)[:max_articles]
        ranked_videos = self._ranker.rank(query, videos)[:max_videos]

        result = {
            "query": query,
            "topic": topic,
            "books": ranked_books,
            "articles": ranked_articles,
            "videos": ranked_videos,
            "timestamp": datetime.now().isoformat(),
        }

        # ── Cache write ──
        if use_cache:
            self._write_cache(cache_key, result)

        return result

    # ── Cache helpers ───────────────────────────────────────
    def _read_cache(self, key: str) -> Optional[Dict]:
        path = CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
            if (datetime.now() - ts).total_seconds() < CACHE_TTL_SECONDS:
                logger.info("Cache hit: %s", key[:8])
                return data
        except Exception:
            pass
        return None

    def _write_cache(self, key: str, data: Dict):
        try:
            (CACHE_DIR / f"{key}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Cache write error: %s", e)


# ─────────────────────────────────────────────────────────────
# FORMAT HELPERS (for text-mode rendering in chat)
# ─────────────────────────────────────────────────────────────

def format_results_markdown(results: Dict[str, Any]) -> str:
    """Format search results as Arabic markdown for the chat interface."""
    parts = ["## 🔎 نتائج البحث\n"]

    topic = results.get("topic", {})
    if topic.get("category"):
        parts.append(f"🧠 **الموضوع:** {topic.get('topic', '')} · **التصنيف:** {topic.get('category', '')}\n")
    parts.append("---\n")

    # Books
    books = results.get("books", [])
    if books:
        parts.append("### 📚 الكتب الموصى بها\n")
        for i, b in enumerate(books, 1):
            authors = "، ".join(b.get("authors", ["مؤلف غير معروف"]))
            score = b.get("relevance_score", 0) * 100
            parts.append(f"**{i}. {b['title']}**")
            parts.append(f"   ✍️ {authors} · 📊 {score:.0f}% تطابق")
            if b.get("description"):
                parts.append(f"   📖 {b['description'][:200]}...")
            if b.get("preview_link"):
                parts.append(f"   🔗 [معاينة الكتاب]({b['preview_link']})")
            parts.append("")

    # Articles
    articles = results.get("articles", [])
    if articles:
        parts.append("### 🌐 مقالات ذات صلة\n")
        for i, a in enumerate(articles, 1):
            score = a.get("relevance_score", 0) * 100
            parts.append(f"**{i}. {a['title']}**")
            parts.append(f"   📰 {a.get('source', 'Wikipedia')} · 📊 {score:.0f}% تطابق")
            if a.get("summary"):
                parts.append(f"   📄 {a['summary'][:250]}...")
            if a.get("url"):
                parts.append(f"   🔗 [اقرأ المقال]({a['url']})")
            parts.append("")

    # Videos
    videos = results.get("videos", [])
    if videos:
        parts.append("### 🎥 فيديوهات مقترحة\n")
        for i, v in enumerate(videos, 1):
            score = v.get("relevance_score", 0) * 100
            views = int(v.get("view_count", 0))
            views_str = f"{views:,}" if views else ""
            parts.append(f"**{i}. {v['title']}**")
            line = f"   📺 {v.get('channel_title', '')} · 📊 {score:.0f}% تطابق"
            if views_str:
                line += f" · 👁️ {views_str} مشاهدة"
            parts.append(line)
            if v.get("url"):
                parts.append(f"   🔗 [شاهد على يوتيوب]({v['url']})")
            parts.append("")

    if not any([books, articles, videos]):
        parts.append("❌ لم يتم العثور على نتائج. حاول إعادة صياغة بحثك.\n")

    parts.append("---")
    parts.append("> ⚠️ هذه النتائج لأغراض معلوماتية فقط. استشر طبيباً مختصاً دائماً.")
    return "\n".join(parts)
