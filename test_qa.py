#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHIFA AI — QA Test Suite
Run with: python -u test_qa.py
"""
import sys, time, traceback

PASS = "\u2705"
FAIL = "\u274c"
results = {}

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

# ─────────────────────────────────────────────────────────────
# TEST 1 — SAFETY LAYER
# ─────────────────────────────────────────────────────────────
section("TEST 1 — SAFETY LAYER")
try:
    from engine.safety import SafetyGuard
    s = SafetyGuard()

    cas1 = "الم شديد في الصدر يمتد الى الذراع الايسر مع تعرق غزير وضيق في التنفس وغثيان مفاجئ"
    cas2 = "صداع شديد في جهة واحدة مع حساسية للضوء"
    cas3 = "تعب عام وخمول بدون سبب واضح"

    r1 = s.detect_emergency(cas1)
    r2 = s.detect_emergency(cas2)
    r3 = s.detect_emergency(cas3)

    ok1 = r1[0] == True
    ok2 = r2[0] == False
    ok3 = r3[0] == False

    print(f"  CAS 1 (cardiac emergency) → expected True  : {PASS if ok1 else FAIL} got={r1[0]}  flags={r1[1]}")
    print(f"  CAS 2 (migraine)          → expected False : {PASS if ok2 else FAIL} got={r2[0]}")
    print(f"  CAS 3 (fatigue generale)  → expected False : {PASS if ok3 else FAIL} got={r3[0]}")

    long_text = "ا" * 1500
    result = s.check(long_text)
    ok4 = result["level"] == "blocked"
    print(f"  Payload >1200 chars bloque → expected blocked: {PASS if ok4 else FAIL} got={result['level']}")

    results["Safety — emergency cardiac"]  = PASS if ok1 else FAIL
    results["Safety — no-FP migraine"]     = PASS if ok2 else FAIL
    results["Safety — no-FP fatigue"]      = PASS if ok3 else FAIL
    results["Safety — payload block"]      = PASS if ok4 else FAIL

except Exception as e:
    print(f"  {FAIL} EXCEPTION: {e}")
    traceback.print_exc()
    results["Safety Layer"] = f"{FAIL} CRASH"

# ─────────────────────────────────────────────────────────────
# TEST 2 — RAG PIPELINE
# ─────────────────────────────────────────────────────────────
section("TEST 2 — RAG PIPELINE")
questions = [
    "ما هي اعراض السكري؟",
    "كيف اعرف اذا كنت اعاني من ضغط الدم؟",
    "ما هو علاج الصداع النصفي؟",
]
docs_last = None
try:
    from engine.retriever import FAISSRetriever
    r = FAISSRetriever()
    t0 = time.time()
    r.load()
    print(f"  Loaded in {time.time()-t0:.2f}s")

    for i, q in enumerate(questions):
        t0 = time.time()
        docs = r.search(q, top_k=3)
        elapsed = time.time() - t0
        limit = 10.0 if i == 0 else 2.0
        ok = elapsed < limit and len(docs) > 0
        docs_last = docs
        print(f"  {PASS if ok else FAIL} [{elapsed:.3f}s] {q[:35]}... → {len(docs)} docs")
        results[f"RAG search — {q[:20]}"] = PASS if ok else FAIL

except Exception as e:
    print(f"  {FAIL} RAG EXCEPTION: {e}")
    traceback.print_exc()
    results["RAG FAISS"] = f"{FAIL} CRASH: {e}"

# ─────────────────────────────────────────────────────────────
# TEST 3 — LLM GROQ
# ─────────────────────────────────────────────────────────────
section("TEST 3 — LLM GROQ")
try:
    from agents.llm_agent import LLMAgent
    llm = LLMAgent()
    ok_health = llm.health_check()
    print(f"  health_check() → {ok_health}")

    if ok_health:
        t0 = time.time()
        resp = llm.run(
            query="ما هي اعراض السكري؟",
            context={
                "kb_context": "السكري من النوع الثاني يسبب زيادة في الشعور بالعطش والتبول وخسارة الوزن",
                "intent": "medical_info",
                "history": [],
            }
        )
        elapsed = time.time() - t0
        lines = len(resp.answer.strip().splitlines()) if resp.success else 0
        chars = len(resp.answer) if resp.success else 0
        ok_time  = elapsed < 5.0
        ok_len   = lines <= 15
        ok_succ  = resp.success
        print(f"  Temps   : {elapsed:.2f}s  {'(OK <5s)' if ok_time else '(SLOW!)'}")
        print(f"  Lignes  : {lines}  {'(OK <=15)' if ok_len else '(TROP!)'}")
        print(f"  Chars   : {chars}")
        print(f"  Success : {resp.success}")
        print(f"  Extrait : {resp.answer[:200]}")
        results["LLM Groq — latence"]  = PASS if ok_time else FAIL
        results["LLM Groq — longueur"] = PASS if ok_len  else FAIL
        results["LLM Groq — succes"]   = PASS if ok_succ else FAIL
    else:
        print(f"  {FAIL} LLM non disponible (cle manquante ou erreur reseau)")
        results["LLM Groq"] = f"{FAIL} NOT AVAILABLE"

except Exception as e:
    print(f"  {FAIL} LLM EXCEPTION: {e}")
    traceback.print_exc()
    results["LLM Groq"] = f"{FAIL} CRASH"

# ─────────────────────────────────────────────────────────────
# TEST 4 — VISION ROUTER
# ─────────────────────────────────────────────────────────────
section("TEST 4 — VISION MODULES")
try:
    import numpy as np
    from PIL import Image
    from engine.vision_router import VisionRouter

    router = VisionRouter()

    skin_img  = Image.fromarray((np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)))
    xray_img  = Image.fromarray((np.random.randint(0,   100, (224, 224, 3), dtype=np.uint8)))
    brain_img = Image.fromarray((np.random.randint(20,  180, (224, 224, 3), dtype=np.uint8)))
    rgba_img  = Image.fromarray((np.random.randint(0,   255, (200, 200, 4), dtype=np.uint8)), "RGBA")
    small_img = Image.fromarray((np.random.randint(0,   255, ( 32,  32, 3), dtype=np.uint8)))

    tests = [
        (skin_img,  "dermato",   "Peau (300x300 RGB)"),
        (xray_img,  "xray",      "X-Ray (224x224 RGB)"),
        (brain_img, "brain_mri", "IRM cerveau (224x224)"),
        (rgba_img,  "dermato",   "RGBA → RGB conversion"),
        (small_img, "xray",      "Petite image 32x32"),
    ]

    for img, vtype, label in tests:
        t0 = time.time()
        try:
            result = router.analyze(img, vtype)
            elapsed = time.time() - t0
            valid = result.get("valid", "?")
            conf  = result.get("confidence", 0)
            cls   = result.get("class", "N/A")
            cam   = result.get("gradcam") is not None
            print(f"  {PASS} {label}")
            print(f"       valid={valid} | conf={conf:.2f} | gradcam={cam} | [{elapsed:.1f}s] | class={cls}")
            results[f"Vision — {label}"] = PASS
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {FAIL} {label} → {type(e).__name__}: {e}")
            results[f"Vision — {label}"] = f"{FAIL} {type(e).__name__}"

except Exception as e:
    print(f"  {FAIL} VISION SETUP EXCEPTION: {e}")
    traceback.print_exc()
    results["Vision Router"] = f"{FAIL} CRASH: {e}"

# ─────────────────────────────────────────────────────────────
# TEST 5 — GEOLOCATION + NEARBY CARE (OpenStreetMap)
# ─────────────────────────────────────────────────────────────
section("TEST 5 — NEARBY CARE (OpenStreetMap)")
try:
    from engine.nearby_care import get_nearby_doctors
    t0 = time.time()
    res = get_nearby_doctors(33.9716, -6.8498, 5000)
    elapsed = time.time() - t0

    ok_time = elapsed < 10.0
    ok_res  = len(res) >= 1

    print(f"  Temps     : {elapsed:.2f}s  {'OK' if ok_time else 'LENT!'}")
    print(f"  Resultats : {len(res)} etablissements")
    for i, h in enumerate(res[:3], 1):
        print(f"  {i}. {h.get('name','?')} | dist={h.get('distance_km','?')}km | tel={h.get('phone','N/A')}")

    results["OpenStreetMap — latence"]    = PASS if ok_time else FAIL
    results["OpenStreetMap — resultats"]  = PASS if ok_res  else f"{FAIL} 0 results"

except Exception as e:
    print(f"  {FAIL} EXCEPTION: {e}")
    traceback.print_exc()
    results["OpenStreetMap"] = f"{FAIL} CRASH: {e}"

# ─────────────────────────────────────────────────────────────
# TEST 6 — TRIAGE CLASSIFIER
# ─────────────────────────────────────────────────────────────
section("TEST 6 — TRIAGE CLASSIFIER")
try:
    from engine.triage import MedicalTriageClassifier, RiskLevel
    tc = MedicalTriageClassifier()

    t_tests = [
        ("لا استطيع التنفس ووجعي في صدري شديد",      RiskLevel.EMERGENCY, "emergency"),
        ("عندي صداع خفيف منذ الصباح",                  RiskLevel.SAFE,      "safe"),
        ("دم في البول وفقدان وزن مفاجئ",               RiskLevel.MODERATE,  "moderate"),
    ]
    for text, expected_risk, label in t_tests:
        t0 = time.time()
        r  = tc.classify(text)
        elapsed = time.time() - t0
        ok = r.risk_level == expected_risk
        print(f"  {PASS if ok else FAIL} [{elapsed:.2f}s] {label}: score={r.score:.2f} risk={r.risk_level.value}")
        results[f"Triage — {label}"] = PASS if ok else f"{FAIL} got={r.risk_level.value}"

except Exception as e:
    print(f"  {FAIL} TRIAGE EXCEPTION: {e}")
    traceback.print_exc()
    results["Triage Classifier"] = f"{FAIL} CRASH"

# ─────────────────────────────────────────────────────────────
# RAPPORT FINAL
# ─────────────────────────────────────────────────────────────
section("RAPPORT FINAL QA")
all_pass = all(str(v).startswith(PASS) for v in results.values())
print(f"\n{'Module':<40} {'Statut'}")
print("-" * 55)
for module, status in results.items():
    print(f"  {module:<40} {status}")

print()
total = len(results)
passed = sum(1 for v in results.values() if str(v).startswith(PASS))
print(f"  Score global : {passed}/{total}  {'🎉 TOUS PASSES' if all_pass else '⚠️  ECHECS DETECTES'}")
sys.exit(0 if all_pass else 1)
