# -*- coding: utf-8 -*-
"""SHIFA AI — Full Audit Scan (Phase 0)"""
import ast, os, sys, re, io

# Fix Windows encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("  SHIFA AI — PHASE 0: SCAN COMPLET")
print("=" * 60)

# ── 1. Syntax errors ──
print("\n[1] Scan de syntaxe Python...")
errors = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'node_modules', '.agent']]
    for f in files:
        if f.endswith('.py'):
            fpath = os.path.join(root, f)
            try:
                with open(fpath, encoding='utf-8') as fh:
                    src = fh.read()
                ast.parse(src)
            except SyntaxError as e:
                errors.append(f'  SYNTAX ERROR {fpath}: {e}')
            except Exception as e:
                errors.append(f'  ERROR {fpath}: {e}')

if errors:
    print("❌ Erreurs trouvées:")
    for e in errors:
        print(e)
else:
    print("✅ Zéro erreur de syntaxe dans tout le projet")

# ── 2. Critical imports ──
print("\n[2] Vérification imports critiques...")
imports = [
    'import torch',
    'import torchvision',
    'import streamlit',
    'import numpy',
    'import PIL',
    'import sklearn',
    'import requests',
]
for imp in imports:
    try:
        exec(imp)
        print(f'  ✅ {imp}')
    except ImportError as e:
        print(f'  ❌ MANQUANT : {imp} → {e}')
    except Exception as e:
        print(f'  ⚠️  {imp} → {e}')

# Optional imports
optional = [
    ('import faiss', 'faiss'),
    ('from groq import Groq', 'groq'),
    ('import monai', 'monai'),
]
for imp, name in optional:
    try:
        exec(imp)
        print(f'  ✅ {imp}')
    except ImportError:
        print(f'  ⚠️  {imp} (optionnel — non installé)')

# ── 3. except:pass scan ──
print("\n[3] Scan except:pass...")
found_pass = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', '.agent']]
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                lines = open(path, encoding='utf-8').readlines()
                for i, l in enumerate(lines, 1):
                    if re.search(r'^\s*except\s*:\s*(pass)?\s*$', l):
                        found_pass.append(f'  {path}:{i} → {l.strip()}')
            except:
                pass

if found_pass:
    print("❌ except:pass trouvés :")
    for f in found_pass:
        print(f)
else:
    print("✅ Zéro except:pass")

# ── 4. weights=None check ──
print("\n[4] Vérification weights dans modèles vision...")
vision_files = ['engine/dermato.py', 'engine/xray.py', 'engine/brain_mri.py']
for vf in vision_files:
    if os.path.exists(vf):
        content = open(vf, encoding='utf-8').read()
        if 'weights=None' in content:
            print(f'  ❌ weights=None trouvé dans {vf}')
        elif 'weights=' in content:
            match = re.search(r'weights=([^\)]+)', content)
            print(f'  ✅ {vf} → weights={match.group(1) if match else "OK"}')
        else:
            print(f'  ℹ️  {vf} → pas de paramètre weights trouvé')
    else:
        print(f'  ⚠️  {vf} n\'existe pas')

# ── 5. API keys exposure ──
print("\n[5] Scan clés API exposées...")
patterns = [
    r'gsk_[a-zA-Z0-9]{40,}',
    r'sk-[a-zA-Z0-9]{40,}',
    r'AIza[a-zA-Z0-9]{35}',
    r'hf_[a-zA-Z0-9]{30,}',
]
found_keys = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.env', 'venv', '.agent']]
    for f in files:
        if f.endswith(('.py', '.md', '.json', '.toml', '.txt')):
            path = os.path.join(root, f)
            try:
                content = open(path, encoding='utf-8').read()
                for p in patterns:
                    if re.search(p, content):
                        found_keys.append(f'  🚨 CLÉ EXPOSÉE dans {path}')
            except:
                pass

if found_keys:
    for fk in found_keys:
        print(fk)
else:
    print("✅ Aucune clé API exposée dans le code")

# ── 6. .gitignore check ──
print("\n[6] Vérification .gitignore...")
critical = ['.env', 'consultation_history.json', '*.pth', '*.bin', '*.pkl', 'models/']
if os.path.exists('.gitignore'):
    gitignore = open('.gitignore').read()
    for item in critical:
        check = item.replace('*', '').strip('/')
        status = '✅' if check in gitignore else '❌'
        print(f'  {status} .gitignore contient : {item}')
else:
    print("  ❌ .gitignore non trouvé!")

# ── 7. Breast module check ──
print("\n[7] Vérification engine/breast.py...")
if os.path.exists('engine/breast.py'):
    try:
        import py_compile
        py_compile.compile('engine/breast.py', doraise=True)
        print("  ✅ engine/breast.py compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ engine/breast.py: {e}")
else:
    print("  ⚠️  engine/breast.py n'existe pas")

# ── 8. nearby_care check ──
print("\n[8] Vérification engine/nearby_care.py...")
if os.path.exists('engine/nearby_care.py'):
    try:
        import py_compile
        py_compile.compile('engine/nearby_care.py', doraise=True)
        print("  ✅ engine/nearby_care.py compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ engine/nearby_care.py: {e}")
else:
    print("  ⚠️  engine/nearby_care.py n'existe pas")

print("\n" + "=" * 60)
print("  PHASE 0 TERMINÉE")
print("=" * 60)
