# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Unified Model Evaluation Script
  
  Evaluates all three segments of the thesis:
    1. Medical BERT (text classification)
    2. Multi-model cancer benchmark (6 algorithms)
    3. Report prioritizer (BERT+Attention+GRU)
  
  Produces:
    - Comparative metrics tables
    - LaTeX-ready tables for thesis
    - JSON results for dashboards
  
  USAGE:
    python evaluate_all.py
    python evaluate_all.py --segment 1  (run only segment 1)
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime


def evaluate_segment1():
    """Segment 1: Medical BERT Classification."""
    print("━" * 60)
    print("  🔵 Segment 1: Medical BERT Evaluation")
    print("━" * 60)

    results = {"segment": 1, "name": "Medical BERT Classification"}

    try:
        from engine.bert_medical import MedicalBERT
        bert = MedicalBERT()

        if bert.load():
            from train_bert_medical import generate_synthetic_data
            texts, labels = generate_synthetic_data()

            eval_results = bert.evaluate_detailed(texts, labels)
            results.update(eval_results)
            results["status"] = "success"

            print(f"  ✅ Accuracy:  {eval_results['accuracy']:.4f}")
            print(f"  ✅ Precision: {eval_results['precision']:.4f}")
            print(f"  ✅ Recall:    {eval_results['recall']:.4f}")
            print(f"  ✅ F1-Score:  {eval_results['f1_score']:.4f}")
        else:
            results["status"] = "model_not_found"
            print("  ⚠️  Model not trained. Run: python train_bert_medical.py")
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"  ❌ Error: {e}")

    return results


def evaluate_segment2():
    """Segment 2: Multi-model Cancer Benchmark."""
    print(f"\n{'━' * 60}")
    print("  🟢 Segment 2: Multi-Model Cancer Benchmark")
    print("━" * 60)

    results_file = "models/benchmark_results/benchmark_results.json"

    if os.path.exists(results_file):
        with open(results_file) as f:
            benchmark = json.load(f)

        print(f"  ✅ Loaded benchmark results ({len(benchmark)} models)")
        best = max(benchmark.keys(), key=lambda m: benchmark[m].get("f1_score", 0))
        print(f"  🏆 Best model: {best} (F1: {benchmark[best]['f1_score']:.4f})")

        return {
            "segment": 2,
            "name": "Multi-Model Cancer Benchmark",
            "status": "success",
            "models": benchmark,
            "best_model": best,
        }
    else:
        print("  ⚠️  Results not found. Run: python benchmark_models.py")
        return {"segment": 2, "status": "not_run"}


def evaluate_segment3():
    """Segment 3: Report Prioritizer (BERT+Attention+GRU)."""
    print(f"\n{'━' * 60}")
    print("  🟡 Segment 3: Report Prioritizer Evaluation")
    print("━" * 60)

    results = {"segment": 3, "name": "BERT+Attention+GRU Report Prioritizer"}

    try:
        from engine.report_prioritizer import ReportPrioritizer
        prioritizer = ReportPrioritizer()

        if prioritizer.load():
            from train_report_prioritizer import generate_training_data
            texts, labels = generate_training_data()

            # Use first 200 for evaluation
            texts = texts[:200]
            labels = labels[:200]

            eval_results = prioritizer.evaluate(texts, labels)
            results.update(eval_results)
            results["status"] = "success"

            print(f"  ✅ Accuracy:  {eval_results['accuracy']:.4f}")
            print(f"  ✅ Precision: {eval_results['precision']:.4f}")
            print(f"  ✅ Recall:    {eval_results['recall']:.4f}")
            print(f"  ✅ F1-Score:  {eval_results['f1_score']:.4f}")
        else:
            # Test rule-based fallback
            print("  📌 BERT model not trained, evaluating rule-based system")
            test_cases = [
                ("أشعة الصدر طبيعية. لا كتل.", 0),
                ("عقدة مشبوهة في الرئة. خزعة مطلوبة.", 1),
                ("نزيف حاد داخل الجمجمة. جراحة طارئة.", 2),
            ]
            correct = 0
            for text, expected in test_cases:
                pred = prioritizer.predict(text)
                is_correct = pred["priority"] == expected
                correct += int(is_correct)
                status = "✅" if is_correct else "❌"
                print(f"  {status} {pred['icon']} {pred['label_ar']} (expected: {expected})")

            results["rule_based_accuracy"] = correct / len(test_cases)
            results["status"] = "rule_based_only"
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        print(f"  ❌ Error: {e}")

    return results


def generate_latex_table(all_results):
    """Generate a LaTeX table for the thesis."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparative Results of All Three Segments}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Model/Segment} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\",
        r"\hline",
    ]

    for result in all_results:
        if result.get("status") == "success":
            name = result.get("name", "Unknown")
            acc = result.get("accuracy", 0)
            prec = result.get("precision", 0)
            rec = result.get("recall", 0)
            f1 = result.get("f1_score", 0)
            lines.append(f"{name} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} \\\\")

    # Add benchmark models if available
    for result in all_results:
        if result.get("segment") == 2 and "models" in result:
            for model_name, m in result["models"].items():
                lines.append(
                    f"{model_name} & {m.get('accuracy',0):.4f} & "
                    f"{m.get('precision',0):.4f} & {m.get('recall',0):.4f} & "
                    f"{m.get('f1_score',0):.4f} \\\\"
                )

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=int, default=0, help="1, 2, or 3 (0=all)")
    parser.add_argument("--output", default="models/evaluation_results")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SHIFA AI — Unified Model Evaluation                      ║")
    print(f"║   Date: {datetime.now().strftime('%Y-%m-%d %H:%M'):<50}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    all_results = []

    if args.segment in (0, 1):
        all_results.append(evaluate_segment1())
    if args.segment in (0, 2):
        all_results.append(evaluate_segment2())
    if args.segment in (0, 3):
        all_results.append(evaluate_segment3())

    # Save results
    os.makedirs(args.output, exist_ok=True)

    output_file = os.path.join(args.output, "all_results.json")
    # Remove non-serializable items
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if k != "confusion_matrix" or not isinstance(v, np.ndarray)}
        clean_results.append(clean)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  💾 Results saved: {output_file}")

    # Generate LaTeX
    latex = generate_latex_table(all_results)
    latex_file = os.path.join(args.output, "results_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex)
    print(f"  📄 LaTeX table saved: {latex_file}")

    print(f"\n{'═' * 60}")
    print("  🎉 Evaluation Complete!")
    print("═" * 60)


if __name__ == "__main__":
    main()
