# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — Report Prioritizer Training Script
  Architecture: BERT + Self-Attention + Bidirectional GRU
  
  Trains the radiology report prioritizer using:
    1. Synthetic training data (Arabic + English radiology reports)
    2. Optional: MIMIC-III / OpenI datasets
  
  USAGE:
    python train_report_prioritizer.py
    python train_report_prioritizer.py --epochs 10 --batch_size 8
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import Counter

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
#  Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════

def generate_training_data():
    """
    Generate synthetic radiology reports for training.
    In production, replace this with real datasets (MIMIC-III, OpenI, etc.)
    """
    data = []

    # ── Priority 0: Routine ──────────────────────────────────────
    routine_reports = [
        "الأشعة السينية للصدر: لا توجد علامات مرضية واضحة. القلب والرئتان طبيعيان. لا يوجد انصباب جنبي.",
        "تصوير الثدي الشعاعي: لا توجد كتل مشبوهة. أنسجة الثدي طبيعية. BI-RADS 1.",
        "أشعة الصدر: الحقول الرئوية صافية. ظل القلب طبيعي الحجم. لا يوجد استرواح صدري.",
        "الرنين المغناطيسي للدماغ: لا توجد آفات داخل الجمجمة. البطينات طبيعية. لا يوجد تحول في خط المنتصف.",
        "تصوير البطن بالموجات فوق الصوتية: الكبد والطحال والكلى طبيعية. لا يوجد سوائل حرة في البطن.",
        "Chest X-ray: Clear lung fields bilaterally. Normal cardiac silhouette. No pleural effusion.",
        "Mammogram: No suspicious masses or calcifications. Normal breast tissue. BI-RADS 1 negative.",
        "Brain MRI: Normal intracranial structures. No evidence of acute infarction or mass lesion.",
        "Abdominal ultrasound: Normal liver, spleen, and kidneys. No free fluid. No gallstones.",
        "CT abdomen: Normal appendix visualized. No acute pathology. Incidental simple renal cyst.",
        "Spine X-ray: Normal vertebral alignment. No fractures. Mild degenerative changes appropriate for age.",
        "Hip X-ray: No fracture or dislocation. Joint spaces preserved. No significant arthropathy.",
        "أشعة العمود الفقري: محاذاة طبيعية. لا كسور. تغيرات تنكسية خفيفة مناسبة للعمر.",
        "أشعة الركبة: لا كسور. المفاصل سليمة. لا علامات على التهاب المفاصل الحاد.",
        "تصوير الغدة الدرقية: حجم طبيعي. لا عقد مشبوهة. تدفق الدم طبيعي.",
    ]

    # ── Priority 1: Semi-Urgent ──────────────────────────────────
    semi_urgent_reports = [
        "أشعة الصدر: عقدة رئوية معزولة في الفص العلوي الأيمن بحجم 8 ملم. يُنصح بمتابعة بالتصوير المقطعي.",
        "تصوير الثدي: كتلة ذات حدود غير واضحة في الثدي الأيسر. BI-RADS 4. يُنصح بأخذ خزعة.",
        "الرنين المغناطيسي: آفة مشبوهة في الفص الجبهي. تعزيز غير متجانس. يُنصح بمراجعة عصبية.",
        "أشعة الصدر: تكثف في القاعدة اليمنى. انصباب جنبي معتدل. قد يمثل التهاب رئوي.",
        "تصوير البطن: تضخم في الكبد مع آفة بؤرية مشبوهة. يُنصح بتصوير مقطعي مع الصبغة.",
        "Chest CT: Solitary pulmonary nodule in the right upper lobe, 12mm. Recommend follow-up in 3 months.",
        "Mammogram: Suspicious cluster of microcalcifications in the left breast. BI-RADS 4B. Biopsy recommended.",
        "Brain MRI: Suspicious enhancing lesion in the left temporal lobe. Further evaluation recommended.",
        "CT abdomen: Enlarged lymph nodes in the mesentery. Largest measuring 2.5cm. Lymphoma cannot be excluded.",
        "Chest X-ray: Right lower lobe consolidation with air bronchograms. Moderate pleural effusion.",
        "CT chest: Multiple bilateral pulmonary nodules. Largest 15mm. Differential includes metastatic disease.",
        "Liver ultrasound: Hepatic lesion with irregular borders. Cannot exclude malignancy. CT recommended.",
        "عقدة رئوية مشبوهة في الفص الأيسر. الحدود غير منتظمة. يُنصح بخزعة موجهة بالأشعة.",
        "تصوير مقطعي للبطن: كتلة في الكلية اليمنى بحجم 4 سم. احتمال ورم كلوي. يُنصح بجراحة.",
        "ارتشاح رئوي منتشر ثنائي الجانب. لا يمكن استبعاد العدوى الانتهازية. يُنصح بزراعة البلغم.",
    ]

    # ── Priority 2: Urgent ───────────────────────────────────────
    urgent_reports = [
        "أشعة الصدر: استرواح صدري توتري في الجانب الأيسر. انحراف القصبة الهوائية والمنصف. تدخل فوري مطلوب.",
        "التصوير المقطعي للدماغ: نزيف داخل الجمجمة حاد. ورم دموي تحت الجافية. تحول في خط المنتصف. جراحة طارئة.",
        "تصوير الأوعية المقطعي: انصمام رئوي ثنائي حاد. تورم في البطين الأيمن. علاج فوري بمضادات التخثر.",
        "أشعة البطن: هواء حر تحت الحجاب الحاجز. احتمال انثقاب في الجهاز الهضمي. تدخل جراحي عاجل.",
        "الرنين المغناطيسي: سكتة دماغية حادة في حوض الشريان الدماغي الأوسط الأيسر. إجراء فوري مطلوب.",
        "CT head: Acute subarachnoid hemorrhage. Hydrocephalus. Urgent neurosurgical consultation required.",
        "CT chest: Massive pulmonary embolism with right heart strain. Immediate anticoagulation required.",
        "CT abdomen: Free air under the diaphragm. Perforated viscus. Urgent surgical consultation.",
        "Chest X-ray: Tension pneumothorax with mediastinal shift. Emergent chest tube placement required.",
        "CT angiography: Acute aortic dissection type A. Emergent surgical repair indicated.",
        "Brain CT: Large intracerebral hemorrhage with midline shift >5mm. Herniation syndrome. Emergent decompression.",
        "CT pelvis: Ruptured abdominal aortic aneurysm with active hemorrhage. Immediate vascular surgery.",
        "نزيف حاد داخل البطين الدماغي. استسقاء دماغي حاد. تحويلة بطينية طارئة مطلوبة.",
        "كسر غير مستقر في العمود الفقري العنقي C2. خطر إصابة الحبل الشوكي. تثبيت فوري مطلوب.",
        "احتشاء عضلة القلب الحاد مع ارتفاع ST. انسداد الشريان التاجي. قسطرة فورية مطلوبة.",
    ]

    for report in routine_reports:
        data.append({"text": report, "priority": 0})
    for report in semi_urgent_reports:
        data.append({"text": report, "priority": 1})
    for report in urgent_reports:
        data.append({"text": report, "priority": 2})

    # Augment with variations
    augmented = []
    prefixes = [
        "تقرير الأشعة: ", "نتائج التصوير: ", "Findings: ", "Impression: ",
        "المعطيات: ", "الخلاصة: ", "Report: ", "Clinical Findings: ",
    ]

    for item in data:
        for prefix in prefixes[:3]:  # 3 augmentations per sample
            augmented.append({
                "text": prefix + item["text"],
                "priority": item["priority"],
            })

    data.extend(augmented)
    np.random.shuffle(data)

    texts = [d["text"] for d in data]
    labels = [d["priority"] for d in data]

    return texts, labels


# ═══════════════════════════════════════════════════════════════════
#  Main Training
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train BERT+Attention+GRU Report Prioritizer")
    parser.add_argument("--model", default="aubmindlab/bert-base-arabertv2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SHIFA AI — Report Prioritizer Training                   ║")
    print("║   Architecture: BERT + Self-Attention + BiGRU              ║")
    print(f"║   Model: {args.model:<49}║")
    print(f"║   Device: {str(DEVICE):<48}║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Step 1: Generate Data ──
    print("━" * 60)
    print("  📦 Step 1: Generating Training Data")
    print("━" * 60)

    texts, labels = generate_training_data()

    print(f"  ✅ Total samples: {len(texts)}")
    print(f"  📊 Distribution: {dict(Counter(labels))}")
    print(f"     🟢 Routine:     {labels.count(0)}")
    print(f"     🟠 Semi-urgent: {labels.count(1)}")
    print(f"     🔴 Urgent:      {labels.count(2)}")

    # ── Step 2: Train ──
    from engine.report_prioritizer import ReportPrioritizer

    prioritizer = ReportPrioritizer(model_name=args.model)

    history = prioritizer.fine_tune(
        texts=texts,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verbose=True,
    )

    # ── Step 3: Save ──
    print(f"\n{'━' * 60}")
    print("  💾 Step 3: Saving Model")
    print("━" * 60)

    prioritizer.save()

    history_path = "models/report_prioritizer/training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✅ History saved: {history_path}")

    # ── Step 4: Evaluate ──
    print(f"\n{'━' * 60}")
    print("  📊 Step 4: Evaluation")
    print("━" * 60)

    # Test on a few examples
    test_reports = [
        "أشعة الصدر: الحقول الرئوية صافية. ظل القلب طبيعي.",
        "CT chest: Suspicious nodule in the right upper lobe. Biopsy recommended.",
        "نزيف حاد داخل الجمجمة مع تحول في خط المنتصف. تدخل جراحي فوري.",
    ]
    expected = [0, 1, 2]

    print("\n  🧪 Sample Predictions:")
    for report, exp in zip(test_reports, expected):
        result = prioritizer.predict(report)
        status = "✅" if result["priority"] == exp else "❌"
        print(f"  {status} {result['icon']} {result['label_ar']} "
              f"(conf: {result['confidence']:.3f}) | Expected: {exp}")
        print(f"     📝 {report[:60]}...")

    # Full evaluation
    eval_results = prioritizer.evaluate(texts[:100], labels[:100])
    print(f"\n  📊 Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"  📊 Precision: {eval_results['precision']:.4f}")
    print(f"  📊 Recall:    {eval_results['recall']:.4f}")
    print(f"  📊 F1-Score:  {eval_results['f1_score']:.4f}")
    print(f"\n{eval_results['classification_report']}")

    total_time = time.time() - start_time
    print("═" * 60)
    print(f"  🎉 Training Complete! Total time: {total_time/60:.1f} minutes")
    print("═" * 60)


if __name__ == "__main__":
    main()
