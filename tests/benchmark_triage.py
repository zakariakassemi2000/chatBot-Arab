# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Triage Classifier Benchmark
  100 annotated test cases → Accuracy, Precision, Recall, F1 per class.
  
  USAGE: python tests/benchmark_triage.py
═══════════════════════════════════════════════════════════════════════
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

from engine.triage import MedicalTriageClassifier, RiskLevel


# ═══════════════════════════════════════════════════════════════════
#  Annotated Dataset: 100 cases (33 Emergency, 34 Moderate, 33 Safe)
# ═══════════════════════════════════════════════════════════════════

ANNOTATED_CASES = [
    # ── EMERGENCY (33 cases) ─────────────────────────────────────
    ("لا أستطيع التنفس", "emergency"),
    ("توقف التنفس عند أخي", "emergency"),
    ("اختناق شديد ولا أستطيع البلع", "emergency"),
    ("ضيق تنفس شديد فجأة", "emergency"),
    ("مش قادر اتنفس", "emergency"),
    ("نوبة قلبية عند أبي", "emergency"),
    ("توقف القلب", "emergency"),
    ("ألم شديد في الصدر مفاجئ", "emergency"),
    ("ألم في الصدر وضيق التنفس معاً", "emergency"),
    ("ألم في الصدر مع تعرق غزير", "emergency"),
    ("فقدت الوعي في العمل", "emergency"),
    ("إغماء مفاجئ", "emergency"),
    ("غيبوبة منذ ساعتين", "emergency"),
    ("فاقد الوعي ولا يستجيب", "emergency"),
    ("نزيف شديد بعد حادث", "emergency"),
    ("نزيف لا يتوقف من الجرح", "emergency"),
    ("نزيف من الرأس بعد سقوط", "emergency"),
    ("شلل مفاجئ في نصف الجسم", "emergency"),
    ("فقدت النطق فجأة", "emergency"),
    ("وجهي مائل ولا أستطيع الكلام", "emergency"),
    ("تنميل نصف الجسم فجأة", "emergency"),
    ("تسمم بمادة كيميائية", "emergency"),
    ("ابتلع سم وبدأ يتقيأ", "emergency"),
    ("شرب كلور بالخطأ", "emergency"),
    ("ابتلع دواء كثير", "emergency"),
    ("حادث سير خطير", "emergency"),
    ("سقوط من ارتفاع عالي", "emergency"),
    ("كسر مفتوح في الساق", "emergency"),
    ("حروق شديدة على كامل الجسم", "emergency"),
    ("جرح عميق ونزيف كثير", "emergency"),
    ("أريد الانتحار", "emergency"),
    ("أفكار انتحارية مستمرة", "emergency"),
    ("بدي اموت ما بقدر اتحمل", "emergency"),

    # ── MODERATE (34 cases) ──────────────────────────────────────
    ("دم في البول منذ يومين", "moderate"),
    ("دم في البراز أحمر فاتح", "moderate"),
    ("فقدان وزن مفاجئ 10 كيلو", "moderate"),
    ("لاحظت كتلة غريبة في رقبتي", "moderate"),
    ("ورم في الإبط", "moderate"),
    ("ارتفاع حرارة شديد 39 درجة", "moderate"),
    ("حرارة أكثر من 39 عند الطفل", "moderate"),
    ("صداع شديد جداً لا يزول", "moderate"),
    ("تشنجات متكررة", "moderate"),
    ("صرع جديد ظهر فجأة", "moderate"),
    ("ألم في الصدر خفيف", "moderate"),
    ("خدر في الذراع اليسرى", "moderate"),
    ("تعرق غزير بالليل", "moderate"),
    ("ألم في الذراع مع تعب", "moderate"),
    ("غثيان مفاجئ ودوخة", "moderate"),
    ("خفقان شديد ونبض سريع", "moderate"),
    ("ضيق التنفس عند المشي", "moderate"),
    ("دوخة شديدة وعدم توازن", "moderate"),
    ("تقيؤ مستمر لا يتوقف", "moderate"),
    ("ألم مستمر في البطن يومين", "moderate"),
    ("ألم حاد في البطن مع حرارة", "moderate"),
    ("سعال مع دم", "moderate"),
    ("ألم شديد في الظهر مفاجئ", "moderate"),
    ("صداع مع تشوش الرؤية", "moderate"),
    ("حرارة مرتفعة مع طفح جلدي", "moderate"),
    ("ألم في العين مع احمرار شديد", "moderate"),
    ("تورم في الوجه مفاجئ", "moderate"),
    ("صعوبة البلع متكررة", "moderate"),
    ("ألم في الأذن مع حرارة", "moderate"),
    ("انتفاخ في الساق مع ألم", "moderate"),
    ("حساسية شديدة مع تورم", "moderate"),
    ("إسهال شديد مع جفاف", "moderate"),
    ("ألم في الكلى حاد", "moderate"),
    ("نبضات سريعة ودوار", "moderate"),

    # ── SAFE (33 cases) ──────────────────────────────────────────
    ("ما هي فوائد شرب الماء؟", "safe"),
    ("أريد معلومات عن التغذية الصحية", "safe"),
    ("كيف أحسن نومي؟", "safe"),
    ("ما هو النظام الغذائي المتوازن؟", "safe"),
    ("نصائح للوقاية من البرد", "safe"),
    ("كم ساعة نوم يحتاج الإنسان؟", "safe"),
    ("ما هي فوائد المشي؟", "safe"),
    ("كيف أخفف الوزن بشكل صحي؟", "safe"),
    ("ما هي فوائد العسل؟", "safe"),
    ("هل القهوة مفيدة للصحة؟", "safe"),
    ("ما هو أفضل وقت للرياضة؟", "safe"),
    ("كيف أقوي مناعتي؟", "safe"),
    ("ما هي فوائد الحليب؟", "safe"),
    ("نصائح للحامل في الشهر الأول", "safe"),
    ("كيف أعتني ببشرتي؟", "safe"),
    ("ما هو علاج حب الشباب؟", "safe"),
    ("كيف أقي نفسي من الشمس؟", "safe"),
    ("ما هي أعراض نقص فيتامين د؟", "safe"),
    ("كيف أتعامل مع الإجهاد؟", "safe"),
    ("ما الفرق بين الزكام والأنفلونزا؟", "safe"),
    ("هل المشي يومياً كافي كرياضة؟", "safe"),
    ("أريد نصائح لصحة القلب", "safe"),
    ("ما هي فوائد الصيام؟", "safe"),
    ("كيف أحمي أسناني؟", "safe"),
    ("ما هو أفضل نظام غذائي للسكري؟", "safe"),
    ("نصائح لتقليل السكر في الأكل", "safe"),
    ("كيف أتجنب آلام الظهر؟", "safe"),
    ("ما هي فوائد التمر؟", "safe"),
    ("كيف أحسن ذاكرتي؟", "safe"),
    ("ما هي أسباب تساقط الشعر؟", "safe"),
    ("هل الشاي الأخضر مفيد؟", "safe"),
    ("ما هي فوائد زيت الزيتون؟", "safe"),
    ("كيف أقلل التوتر؟", "safe"),
]


def run_benchmark():
    """Execute the triage benchmark and compute metrics."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SHIFA AI — Triage Classifier Benchmark                   ║")
    print(f"║   Date: {datetime.now().strftime('%Y-%m-%d %H:%M'):<50}║")
    print(f"║   Cases: {len(ANNOTATED_CASES):<49}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    clf = MedicalTriageClassifier()
    
    y_true = []
    y_pred = []
    errors = []
    
    start = time.time()
    
    for msg, expected in ANNOTATED_CASES:
        result = clf.classify(msg)
        predicted = result.risk_level.value
        y_true.append(expected)
        y_pred.append(predicted)
        
        if predicted != expected:
            errors.append({
                "message": msg,
                "expected": expected,
                "predicted": predicted,
                "score": result.score,
                "flags": result.flags[:3],
            })
    
    total_time = time.time() - start
    avg_time_ms = (total_time / len(ANNOTATED_CASES)) * 1000

    # ── Compute Metrics ──
    classes = ["emergency", "moderate", "safe"]
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    
    # Per-class metrics
    metrics = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # Confusion matrix
    confusion = {}
    for actual in classes:
        confusion[actual] = {}
        for predicted_cls in classes:
            confusion[actual][predicted_cls] = sum(
                1 for t, p in zip(y_true, y_pred) if t == actual and p == predicted_cls
            )

    # ── CRITICAL: False Negative Rate for Emergency ──
    emergency_fn = metrics["emergency"]["fn"]
    emergency_total = sum(1 for t in y_true if t == "emergency")
    emergency_fnr = emergency_fn / emergency_total if emergency_total > 0 else 0

    # ── Print Results ──
    print("━" * 60)
    print(f"  📊 OVERALL ACCURACY: {accuracy:.1%} ({correct}/{len(y_true)})")
    print(f"  ⏱️  Avg inference time: {avg_time_ms:.2f} ms/query")
    print("━" * 60)
    
    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "─" * 60)
    for cls in classes:
        m = metrics[cls]
        emoji = {"emergency": "🔴", "moderate": "🟡", "safe": "🟢"}[cls]
        print(f"  {emoji} {cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} "
              f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    print(f"\n  🚨 Emergency False Negative Rate: {emergency_fnr:.1%} ({emergency_fn}/{emergency_total})")
    if emergency_fnr > 0:
        print("  ⚠️  WARNING: Some emergencies were missed!")
    else:
        print("  ✅ All emergencies correctly detected!")

    # Confusion Matrix
    print(f"\n  📋 Confusion Matrix:")
    print(f"  {'':>15} {'→ emergency':>12} {'→ moderate':>12} {'→ safe':>12}")
    for actual in classes:
        row = "  " + f"{'↓ ' + actual:<15}"
        for pred_cls in classes:
            row += f" {confusion[actual][pred_cls]:>11}"
        print(row)

    # Errors
    if errors:
        print(f"\n  ❌ Misclassified ({len(errors)}):")
        for e in errors[:10]:
            print(f"    • [{e['expected']}→{e['predicted']}] \"{e['message'][:50]}\" (score={e['score']:.3f})")

    # ── Save Results ──
    output_dir = Path("models/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "benchmark": "triage_classifier",
        "date": datetime.now().isoformat(),
        "total_cases": len(ANNOTATED_CASES),
        "accuracy": round(accuracy, 4),
        "avg_inference_ms": round(avg_time_ms, 2),
        "emergency_fnr": round(emergency_fnr, 4),
        "per_class": {cls: {k: round(v, 4) if isinstance(v, float) else v 
                           for k, v in m.items()} for cls, m in metrics.items()},
        "confusion_matrix": confusion,
        "errors": errors,
    }
    
    output_file = output_dir / "triage_benchmark.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n  💾 Results saved: {output_file}")
    print("═" * 60)
    
    return results


if __name__ == "__main__":
    run_benchmark()
