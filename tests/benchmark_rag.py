# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — RAG Pipeline Benchmark
  30 annotated Arabic medical queries → Precision@K, MRR, latency.
  
  USAGE: python tests/benchmark_rag.py
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  30 Annotated Queries with Expected Keywords
# ═══════════════════════════════════════════════════════════════════

ANNOTATED_QUERIES = [
    # (query, list_of_keywords_that_should_appear_in_top5_answers)
    ("ما هي أعراض السكري؟", ["سكري", "عطش", "تبول", "glucose"]),
    ("أعراض ارتفاع ضغط الدم", ["ضغط", "صداع", "دوخة"]),
    ("كيف أعالج الصداع؟", ["صداع", "مسكن", "راحة"]),
    ("ما هي فوائد الرياضة؟", ["رياضة", "صحة", "قلب"]),
    ("أسباب آلام الظهر", ["ظهر", "ألم", "عمود"]),
    ("ما هو الكولسترول؟", ["كولسترول", "دهون", "دم"]),
    ("أعراض الأنفلونزا", ["أنفلونزا", "حرارة", "سعال"]),
    ("كيف أقوي المناعة؟", ["مناعة", "فيتامين", "تغذية"]),
    ("ما هو مرض الربو؟", ["ربو", "تنفس", "حساسية"]),
    ("أسباب تساقط الشعر", ["شعر", "تساقط", "فيتامين"]),
    ("علاج الإمساك طبيعياً", ["إمساك", "ألياف", "ماء"]),
    ("أعراض نقص فيتامين د", ["فيتامين", "عظام", "شمس"]),
    ("ما هو مرض القلب؟", ["قلب", "شرايين", "ضغط"]),
    ("كيف أتعامل مع القلق؟", ["قلق", "توتر", "استرخاء"]),
    ("أسباب الأرق وعلاجه", ["أرق", "نوم", "استرخاء"]),
    ("ما هي أعراض الحمل؟", ["حمل", "غثيان", "دورة"]),
    ("أسباب ألم المعدة", ["معدة", "ألم", "هضم"]),
    ("كيف أحافظ على الكلى؟", ["كلى", "ماء", "ملح"]),
    ("ما هو مرض السكري من النوع الثاني؟", ["سكري", "أنسولين", "وزن"]),
    ("أعراض التهاب المفاصل", ["مفاصل", "التهاب", "ألم"]),
    ("كيف أخفض الحرارة؟", ["حرارة", "مسكن", "ماء"]),
    ("ما هو فقر الدم؟", ["فقر", "حديد", "هيموجلوبين"]),
    ("أسباب الدوخة المفاجئة", ["دوخة", "ضغط", "أذن"]),
    ("كيف أتعامل مع الاكتئاب؟", ["اكتئاب", "نفسي", "علاج"]),
    ("ما هي أضرار التدخين؟", ["تدخين", "رئة", "سرطان"]),
    ("أعراض حساسية الطعام", ["حساسية", "طعام", "طفح"]),
    ("كيف أعالج الزكام في البيت؟", ["زكام", "سوائل", "راحة"]),
    ("ما هو التهاب الحلق؟", ["حلق", "التهاب", "بكتيريا"]),
    ("أسباب التعب المستمر", ["تعب", "نوم", "فيتامين"]),
    ("كيف أحمي نفسي من فيروس كورونا؟", ["كورونا", "كمامة", "لقاح"]),
]


def run_benchmark():
    """Execute the RAG pipeline benchmark."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SHIFA AI — RAG Pipeline Benchmark                        ║")
    print(f"║   Date: {datetime.now().strftime('%Y-%m-%d %H:%M'):<50}║")
    print(f"║   Queries: {len(ANNOTATED_QUERIES):<47}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Try to load retrievers ──
    try:
        from engine.retriever import FAISSRetriever
        retriever = FAISSRetriever()
        loaded = retriever.load()
        if not loaded:
            print("  ⚠️  FAISS index not found. Run setup.py first.")
            return None
    except Exception as e:
        print(f"  ❌ Could not load retriever: {e}")
        return None
    
    hit_at_1 = 0
    hit_at_5 = 0
    reciprocal_ranks = []
    latencies = []
    detailed_results = []

    for query, expected_keywords in ANNOTATED_QUERIES:
        start = time.time()
        results = retriever.search(query, top_k=5)
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        # Check if any expected keyword appears in top-K answers
        answers_text = " ".join([r.get("answer", "") for r in results])
        
        # Hit@1: first result contains at least one expected keyword
        first_answer = results[0].get("answer", "") if results else ""
        hit1 = any(kw in first_answer for kw in expected_keywords)
        if hit1:
            hit_at_1 += 1

        # Hit@5: any of top-5 results contain at least one expected keyword
        hit5 = any(kw in answers_text for kw in expected_keywords)
        if hit5:
            hit_at_5 += 1

        # Reciprocal Rank
        rr = 0.0
        for i, r in enumerate(results):
            if any(kw in r.get("answer", "") for kw in expected_keywords):
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)

        detailed_results.append({
            "query": query,
            "hit_at_1": hit1,
            "hit_at_5": hit5,
            "rr": round(rr, 4),
            "latency_ms": round(latency, 2),
            "top_score": round(results[0]["score"], 4) if results else 0,
        })

    # ── Aggregate Metrics ──
    n = len(ANNOTATED_QUERIES)
    precision_at_1 = hit_at_1 / n
    precision_at_5 = hit_at_5 / n
    mrr = sum(reciprocal_ranks) / n
    avg_latency = sum(latencies) / n

    # ── Print Results ──
    print("━" * 60)
    print(f"  📊 Precision@1:   {precision_at_1:.1%} ({hit_at_1}/{n})")
    print(f"  📊 Precision@5:   {precision_at_5:.1%} ({hit_at_5}/{n})")
    print(f"  📊 MRR:           {mrr:.4f}")
    print(f"  ⏱️  Avg Latency:  {avg_latency:.1f} ms")
    print(f"  ⏱️  P95 Latency:  {sorted(latencies)[int(0.95*n)]:.1f} ms")
    print("━" * 60)

    # Quality breakdown
    print(f"\n  📋 Per-Query Results:")
    for r in detailed_results:
        status = "✅" if r["hit_at_1"] else ("🟡" if r["hit_at_5"] else "❌")
        print(f"    {status} \"{r['query'][:40]}\" → P@1={'Y' if r['hit_at_1'] else 'N'} "
              f"P@5={'Y' if r['hit_at_5'] else 'N'} RR={r['rr']:.2f} ({r['latency_ms']:.0f}ms)")

    # ── Save ──
    output_dir = Path("models/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        "benchmark": "rag_pipeline",
        "date": datetime.now().isoformat(),
        "total_queries": n,
        "precision_at_1": round(precision_at_1, 4),
        "precision_at_5": round(precision_at_5, 4),
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "details": detailed_results,
    }
    
    output_file = output_dir / "rag_benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n  💾 Results saved: {output_file}")
    print("═" * 60)
    return results_data


if __name__ == "__main__":
    run_benchmark()
