import os, sys, ast, re, time
sys.stdout.reconfigure(encoding='utf-8')

print("=== 🔍 PHASE 0: SCAN DE SYNTAXE & IMPORTS ===")
files = []
for root, dirs, fs in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__','.git','venv','node_modules','.venv']]
    for f in fs:
        if f.endswith('.py'):
            files.append(os.path.join(root, f))

# Syntax Errors
print("\n[1] Analyse de la syntaxe Python:")
missing = []
for fpath in files:
    try:
        with open(fpath, encoding='utf-8') as f:
            src = f.read()
        ast.parse(src)
    except SyntaxError as e:
        missing.append(f'❌ SYNTAX ERROR dans {fpath}: {e}')
    except Exception as e:
        missing.append(f'❌ ERROR dans {fpath}: {e}')

if missing:
    for m in missing: print(m)
else:
    print('✅ Zéro erreur de syntaxe dans tout le projet')

# Except pass
print("\n[2] Analyse des except: pass (mauvaises pratiques):")
found = []
for fpath in files:
    try:
        lines = open(fpath, encoding='utf-8').readlines()
        for i,l in enumerate(lines, 1):
            if re.search(r'^\s*except\s*:\s*(pass)?\s*$', l) or re.search(r'^\s*except.*:\s*pass\s*$', l):
                found.append(f'{fpath}:{i} → {l.strip()}')
    except: pass
if found:
    for f in found: print(f'❌ {f}')
else:
    print('✅ Zéro except:pass')

print("\n=== 🛡️ PHASE 2: AUDIT DE SÉCURITÉ ===")
patterns = [
    r'gsk_[a-zA-Z0-9]{40,}',    # Groq
    r'sk-[a-zA-Z0-9]{40,}',      # OpenAI
]
found_keys = []
for root, dirs, fs in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__','.git','venv']]
    for f in fs:
        if f.endswith(('.py','.md','.json','.toml','.txt')) and f != '.env' and 'run_audit' not in f:
            path = os.path.join(root,f)
            try:
                content = open(path, encoding='utf-8').read()
                for p in patterns:
                    if re.search(p, content):
                        found_keys.append(f'🚨 CLÉ API EXPOSÉE dans {path}')
            except: pass
if found_keys:
    for f in list(set(found_keys)): print(f)
else:
    print('✅ Aucune clé API exposée dans le code')
