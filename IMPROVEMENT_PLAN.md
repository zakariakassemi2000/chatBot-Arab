# SHIFA AI — Plan d'amélioration technique (Audit)

> Document d'audit produit pour l'assistant médical SHIFA AI.  
> Statut: certaines améliorations déjà appliquées ✅ | d'autres en attente ⏳

---

## 1. Executive Summary

Le projet **SHIFA AI** est une application Streamlit combinant RAG (FAISS + embeddings), LLM Groq, garde-fous médicaux, STT/TTS et modèles d'imagerie (MRI, X-Ray, densité mammaire). L'architecture de base est solide, mais plusieurs risques en **sécurité**, **déploiement** et **maintenabilité** ont été identifiés.

**Priorités court terme** : sécuriser les secrets, stabiliser le démarrage, améliorer l'observabilité, puis enrichir (RAG, Whisper, dermatologie).

---

## 2. Security Risks

| Risque | Impact | Statut | Action |
|--------|--------|--------|--------|
| `.env` au root | Fuite GROQ_API_KEY / HF_TOKEN si commité | ✅ Mitigé | `.env` dans `.gitignore` |
| Données patient (consultation_history.json) | PHI en clair, pas de chiffrement | ⏳ À faire | Chiffrement au repos, rétention |
| Supply chain (modèles HF) | Versions mouvantes, torch.load (pickle) | ⏳ À faire | Pin revision HF, checksums |
| Logs contenant du texte patient | Exposition en prod | ⏳ À faire | Ne jamais logger contenu patient |

**Bonnes pratiques recommandées** :
- Rotation des clés si `.env` a déjà été commité
- Scanner secrets en CI (gitleaks) — workflow GitHub Actions ajouté
- Minimiser la persistance par défaut ; chiffrer si nécessaire

---

## 3. Architecture Issues

| Problème | Impact | Statut | Action |
|----------|--------|--------|--------|
| Monolithe `app.py` | Difficile à tester, maintenir | ⏳ À faire | Découper en `pages/`, `services/` |
| Config fragmentée | Chemins hardcodés | ⏳ À faire | Centraliser dans `utils/config.py` |
| Historique local JSON | Non scalable multi-user | ⏳ À faire | SQLite puis Postgres |

**Architecture cible recommandée** :

```
shifa/
  ui/pages/          # chat, scanner, imaging/*, history, database
  services/          # rag, llm, safety, audio, imaging
  domain/            # schemas, policies
  infra/             # settings, storage, clients
engine/              # (migration progressive)
utils/
tests/
```

---

## 4. Performance Problems

| Problème | Impact | Statut | Action |
|----------|--------|--------|--------|
| Hybrid search O(n) | Latence si KB grande | ✅ Corrigé | IDs stables, RRF sans remapping texte |
| Charge mémoire au démarrage | Lenteur | ✅ Partiel | Lazy-load brain/xray, cache_resource |
| Warmup non géré | Première requête lente | ⏳ Optionnel | Warmup en background |

---

## 5. Deployment Problems

| Problème | Impact | Statut | Action |
|----------|--------|--------|--------|
| `requirements.txt` monolithique | Installation lente, conflits | ✅ Corrigé | `requirements/base.txt`, `audio.txt`, `imaging.txt`, `dev.txt` |
| KB manquante → crash | Expérience "ça marche pas" | ✅ Corrigé | UI Setup + bouton build KB |
| Docker sans profils | Image trop lourde | ⏳ Optionnel | Build args pour imaging |

**Commandes d'installation** (documentées dans README) :
```bash
pip install -r requirements/base.txt                    # Chat only
pip install -r requirements/base.txt -r requirements/audio.txt
pip install -r requirements/base.txt -r requirements/audio.txt -r requirements/imaging.txt
```

---

## 6. UX / Product Issues

| Problème | Impact | Statut | Action |
|----------|--------|--------|--------|
| Pages imagerie perçues comme diagnostic | Risque légal, mauvaise décision | ⏳ Partiel | Wording "éducatif", limites visibles |
| Restrictions dosage contournables | Bypass par reformulation | ✅ Renforcé | Patterns regex élargis dans SafetyGuard |
| Historique non scalable | Multi-user impossible | ⏳ À faire | SQLite/Postgres |

---

## 7. Concrete Fixes (déjà appliqués ou à appliquer)

### 7.1 Secrets — ✅ Appliqué
- `.env` dans `.gitignore`
- `.env.example` comme modèle
- CI gitleaks (`.github/workflows/ci.yml`)

### 7.2 Dépendances — ✅ Appliqué
- Split `requirements/base.txt`, `audio.txt`, `imaging.txt`, `dev.txt`
- `requirements.txt` inclut les 3 profils (legacy full install)

### 7.3 KB fragile — ✅ Appliqué
- Si FAISS manquant : panneau Setup dans l'UI avec bouton "إعداد قاعدة المعرفة"
- Alternative CLI : `python setup.py`

### 7.4 Retrieval O(n) — ✅ Appliqué
- `engine/retriever.py` : `search()` retourne `"id": idx` ; `hybrid_search()` utilise les IDs directement (plus de boucle texte)

### 7.5 Observabilité — ✅ Partiel
- `utils/logger.py` utilisé dans `app.py` pour TTS, STT, pipeline chat
- Remplacer les `except: pass` restants par `logger.warning/error`

### 7.6 Safety — ✅ Renforcé
- `engine/safety.py` : patterns dosage/diagnostic élargis (variantes linguistiques)
- `POST_LLM_FORBIDDEN` étendu

### 7.7 Tests — ✅ Appliqué
- `RUN_HEAVY_TESTS=0` par défaut ; tests imagerie skippés en CI
- `pytest -q` stable sans modèles lourds

### 7.8 Documentation — ✅ Appliqué
- README complet : installation, `.env`, Docker, setup KB, troubleshooting

---

## 8. Recommended Project Architecture

```
app.py                    # Bootstrap Streamlit (minimal)
shifa/
  ui/
    pages/
      chat.py
      scanner.py
      imaging/
        brain_mri.py
        xray.py
        breast_density.py
        derm.py           # (futur)
      history.py
      database.py
    components/
      layout.py
      widgets.py
  services/
    rag_service.py
    llm_service.py
    safety_service.py
    audio_service.py      # + Whisper (futur)
    imaging_service.py
  domain/
    schemas.py
    policies.py
  infra/
    settings.py
    storage/
      history_repo.py
    clients/
      groq_client.py
      hf_client.py
engine/
utils/
data/
tests/
requirements/
Dockerfile
docker-compose.yml
README.md
IMPROVEMENT_PLAN.md
```

---

## 9. Step-by-Step Improvement Roadmap

### High Priority (sécurité + fiabilité)
- [x] H1 — Secrets : `.env` hors git, gitleaks CI
- [x] H2 — Dépendances : split requirements
- [x] H3 — KB robuste : fallback + bouton Setup
- [x] H4 — Documentation : README complet
- [x] H5 — Safety : patterns dosage/diagnostic renforcés

### Medium Priority (perf + observabilité + CI)
- [x] M1 — Fix hybrid retrieval O(n)
- [x] M2 — Observabilité : logger structuré
- [x] M3 — Tests/CI : marquage slow, pipeline stable

### Low Priority (scaling produit)
- [ ] L1 — Refactor architecture (split app.py)
- [ ] L2 — Storage multi-user (SQLite/Postgres)
- [ ] L3 — Gouvernance modèles (pin HF revision, checksums)

### Enrichissement (nouveaux modèles)
- [x] E1 — RAG : datasets HF supplémentaires (arbml/CQA_MD_ar, MedQA), embeddings paramétrables (EMBED_MODEL)
- [x] E2 — Whisper : STT local (openai/whisper-small) via USE_WHISPER=1
- [x] E3 — Dermatologie : page derm_scanner (ahmed-ai/skin_lesions_classifier, HAM10000)

---

## 10. Production-Ready Deployment

| Approche | Usage |
|----------|-------|
| **Docker simple** | `docker-compose up` — image full |
| **Docker 2 images** | `shifa-web` (Streamlit) + `shifa-worker` (build KB, imaging) |
| **Enterprise** | Streamlit front + API FastAPI backend (RAG/LLM/imaging) |

---

## 11. Security Best Practices (AI Healthcare)

- **Données** : minimisation, chiffrement au repos, rétention, suppression à la demande
- **Audit** : logs d'accès (sans contenu patient), traçabilité versions modèles
- **Garde-fous** : refus diagnostic, escalade urgence, incertitude visible
- **Supply chain** : pin versions, pip-audit, verrouillage modèles HF
- **Accès** : auth si multi-user, rate limiting, anti-abus

---

*Dernière mise à jour : mars 2025*
