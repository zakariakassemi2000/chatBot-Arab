"""SHIFA AI — Full project audit script."""
import ast, os, re, sys, time, io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  SHIFA AI — AUDIT COMPLET (Phase 0-3)")
print("=" * 60)

# ── PHASE 0.1 — Syntax Errors ──
print("\n[Phase 0.1] Scan de syntaxe Python...")
syntax_errors = []
for root, dirs, fs in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "venv", ".agent", "tessdata"]]
    for f in fs:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath, encoding="utf-8") as fh:
                    ast.parse(fh.read())
            except SyntaxError as e:
                syntax_errors.append(f"  SYNTAX ERROR {fpath}: {e}")
            except Exception as e:
                syntax_errors.append(f"  ERROR {fpath}: {e}")

if syntax_errors:
    for m in syntax_errors:
        print(m)
else:
    print("  ✅ Zéro erreur de syntaxe dans tout le projet")

# ── PHASE 0.2 — Critical Imports ──
print("\n[Phase 0.2] Vérification des imports critiques...")
imports = [
    "import torch",
    "import torchvision",
    "import streamlit",
    "import faiss",
    "import numpy",
    "import PIL",
    "import sklearn",
    "import requests",
    "from groq import Groq",
    "import pandas",
    "from rapidfuzz import fuzz",
    "from bs4 import BeautifulSoup",
    "from pydantic import BaseModel",
    "from dotenv import load_dotenv",
]
for imp in imports:
    try:
        exec(imp)
        print(f"  ✅ {imp}")
    except ImportError as e:
        print(f"  ❌ MANQUANT : {imp} → {e}")
    except Exception as e:
        print(f"  ⚠️  {imp} → {e}")

# ── PHASE 0.3 — except:pass scan ──
print("\n[Phase 0.3] Scan except:pass...")
bare_excepts = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "venv", ".agent"]]
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            try:
                lines = open(path, encoding="utf-8").readlines()
                for i, l in enumerate(lines, 1):
                    if re.search(r"^\s*except\s*:\s*(pass)?\s*$", l):
                        bare_excepts.append(f"  {path}:{i} → {l.strip()}")
            except:
                pass
if bare_excepts:
    print("  ❌ except:pass trouvés :")
    for f in bare_excepts:
        print(f)
else:
    print("  ✅ Zéro except:pass")

# ── PHASE 0.4 — API key exposure ──
print("\n[Phase 0.4] Scan clés API exposées...")
patterns = [
    r"gsk_[a-zA-Z0-9]{40,}",
    r"sk-[a-zA-Z0-9]{40,}",
    r"AIza[a-zA-Z0-9]{35}",
    r"hf_[a-zA-Z0-9]{30,}",
]
api_found = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".env", "venv", ".agent"]]
    for f in files:
        if f.endswith((".py", ".md", ".json", ".toml", ".txt")):
            path = os.path.join(root, f)
            try:
                content = open(path, encoding="utf-8").read()
                for p in patterns:
                    if re.search(p, content):
                        api_found.append(f"  🚨 CLÉ EXPOSÉE dans {path}")
            except:
                pass
if api_found:
    for f in api_found:
        print(f)
else:
    print("  ✅ Aucune clé API exposée dans le code")

# ── PHASE 0.5 — .env protection ──
print("\n[Phase 0.5] Protection .env et .gitignore...")
if os.path.exists(".gitignore"):
    gi = open(".gitignore").read()
    checks = [".env", "*.pkl", "*.bin", "consultation_history.json"]
    for item in checks:
        key = item.replace("*", "")
        status = "✅" if key in gi else "❌"
        print(f"  {status} .gitignore contient : {item}")
else:
    print("  ❌ .gitignore introuvable !")

# ── PHASE 1 — Performance ──
print("\n" + "=" * 60)
print("[Phase 1] AUDIT PERFORMANCE")
print("=" * 60)

print("\n[1.1] Chargement FAISS index...")
start = time.time()
try:
    from engine.retriever import HybridRetriever
    r = HybridRetriever(enable_reranker=False)
    loaded = r.load()
    elapsed = time.time() - start
    if loaded:
        status = "✅" if elapsed < 3 else "⚠️ LENT"
        print(f"  {status} FAISS chargé en {elapsed:.2f}s")
    else:
        print(f"  ⚠️ Index FAISS non trouvé (première exécution?)")
except Exception as e:
    print(f"  ❌ FAISS erreur : {e}")

print("\n[1.2] Safety layer performance...")
try:
    from engine.safety import SafetyGuard
    s = SafetyGuard()
    test_texts = [
        "ألم في الصدر مع ضيق في التنفس",
        "صداع خفيف",
    ]
    for txt in test_texts:
        start = time.time()
        result = s.check(txt)
        elapsed = time.time() - start
        level = result.get("level", "?")
        print(f"  ✅ {elapsed*1000:.1f}ms — [{level}] {txt[:40]}")
except Exception as e:
    print(f"  ❌ Safety erreur : {e}")

print("\n[1.3] Import modules vision...")
vision_modules = [
    ("engine.dermato", "DermatoModel"),
    ("engine.xray", "XRayModel"),
    ("engine.brain_mri", "BrainTumorKerasDetector"),
    ("engine.cancer", "CancerDetectorTF"),
    ("engine.breast", "BreastDensityDetector"),
]
for module, cls in vision_modules:
    start = time.time()
    try:
        m = __import__(module, fromlist=[cls])
        getattr(m, cls)
        elapsed = time.time() - start
        print(f"  ✅ {module}.{cls} importé en {elapsed:.2f}s")
    except Exception as e:
        err_short = str(e).split('\n')[0][:80]
        print(f"  ❌ {module} : {err_short}")

# ── SUMMARY ──
print("\n" + "=" * 60)
print("  RÉSUMÉ")
print("=" * 60)
total_issues = len(syntax_errors) + len(bare_excepts) + len(api_found)
print(f"  Erreurs de syntaxe  : {len(syntax_errors)}")
print(f"  except:pass         : {len(bare_excepts)}")
print(f"  Clés API exposées   : {len(api_found)}")
print(f"  TOTAL PROBLÈMES     : {total_issues}")
if total_issues == 0:
    print("  STATUT : 🟢 READY")
else:
    print("  STATUT : 🟡 CORRECTIONS NÉCESSAIRES")
print("=" * 60)
