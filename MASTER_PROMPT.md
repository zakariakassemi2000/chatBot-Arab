# 🏥 SHIFA AI — MASTER PROJECT PROMPT
> Version 1.0 · Plateforme Médicale IA Arabe · Diagnostic · Vision · RAG · Audio
> À utiliser comme contexte système dans tout IDE IA (Google IDX Antigravity, Cursor, GitHub Copilot, etc.)

---

## ██ SECTION 0 — COMMENT UTILISER CE PROMPT

Ce document est le **contexte maître** du projet SHIFA AI.
Il doit être fourni en entier à l'IA assistante avant toute session de développement.
Il décrit : la vision produit, les principes d'ingénierie, l'architecture complète,
les modules existants, les modules en cours, les standards de code,
et les décisions techniques déjà prises.

L'IA doit :
- Respecter scrupuleusement les conventions déjà établies
- Ne jamais proposer de recréer ce qui existe déjà
- Toujours s'intégrer dans la structure modulaire définie
- Prioriser la lisibilité, la maintenabilité et la sécurité médicale
- Répondre en français ou en arabe selon le contexte demandé

---

## ██ SECTION 1 — VISION PRODUIT

### 1.1 Nom et Identité
- **Nom** : SHIFA AI (شفاء الذكاء الاصطناعي)
- **Signification** : "Guérison par l'intelligence artificielle"
- **Langue principale** : Arabe (dialecte marocain + arabe standard moderne)
- **Public cible** : Médecins, infirmiers, patients, professionnels de santé au Maroc et dans le monde arabe

### 1.2 Mission
Fournir une plateforme d'assistance médicale intelligente qui :
1. Permet le diagnostic différentiel basé sur les symptômes décrits en arabe naturel
2. Analyse les images médicales (peau, poumons, cerveau) par deep learning
3. Offre une interface conversationnelle enrichie par une base de connaissances médicale locale (RAG)
4. Fonctionne partiellement hors ligne pour les zones à connectivité limitée
5. Respecte les contraintes éthiques et légales de l'IA médicale

### 1.3 Philosophie Produit
- **Accessibilité** : Interface en arabe, simple, utilisable sur mobile
- **Transparence** : Chaque diagnostic affiche son niveau de confiance et ses limites
- **Sécurité** : Le système ne remplace jamais le médecin — il l'assiste
- **Modularité** : Chaque fonctionnalité est un module indépendant et testable
- **Robustesse** : Fallback offline si l'API cloud est indisponible

---

## ██ SECTION 2 — STACK TECHNIQUE (DÉCISIONS FIGÉES)

> Ces choix sont définitifs. Ne pas proposer d'alternatives sauf demande explicite.

| Couche              | Technologie              | Raison du choix                                      |
|---------------------|--------------------------|------------------------------------------------------|
| Frontend            | Streamlit (Python)       | Prototypage rapide, intégration native avec PyTorch  |
| UI Styling          | CSS custom (Dark Mode + Glassmorphism) | Identité visuelle SHIFA AI          |
| Backend / Logique   | Python 3.11+             | Écosystème IA/ML dominant                            |
| Deep Learning       | PyTorch 2.3+             | Flexibilité, debugging, communauté médicale          |
| Medical Imaging     | MONAI 1.3+               | Transforms médicaux spécialisés, standard industrie  |
| LLM Inference       | Groq API (LLaMA-3)       | Latence ultra-faible (<1s), gratuit generous tier    |
| LLM Fallback        | OpenAI GPT-4o            | Fiabilité si Groq indisponible                       |
| Vector Store        | FAISS (local)            | Offline-first, pas de dépendance cloud pour le RAG   |
| NLP Arabe           | AraBERT (CAMeL-Lab)      | Meilleur modèle pour l'arabe médical                 |
| Orchestration ML    | Scikit-learn             | Classification symptômes, triage                     |
| Speech-to-Text      | OpenAI Whisper (local)   | Support arabe dialectal, offline capable             |
| Text-to-Speech      | gTTS                     | Synthèse vocale arabe simple et légère               |
| Explainability      | Grad-CAM (custom utils/) | Visualisation des zones d'activation pour vision     |
| Environnement       | Google IDX (Antigravity) | IDE cloud, Python natif, déploiement direct          |
| Déploiement         | Streamlit Cloud / HuggingFace Spaces | Gratuit, scalable, GPU disponible      |

---

## ██ SECTION 3 — ARCHITECTURE SYSTÈME

### 3.1 Vue d'ensemble des couches

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (app.py)                        │
│  Streamlit · CSS Dark Mode · Glassmorphism · RTL Arabic UI   │
│  Tabs: Chat | Symptômes | Vision | Audio | Calculateurs      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  SAFETY LAYER (safety.py)                    │
│  Filtrage inputs · Détection intentions nuisibles            │
│  Ajout disclaimer médical · Sanitisation outputs             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼──────────┐       ┌──────────▼──────────┐
│   ENGINE/LLM     │       │   ENGINE/VISION      │
│  llm.py          │       │  vision_router.py    │
│  retriever.py    │       │  dermato.py          │
│  prompt_builder  │       │  xray.py             │
│  FAISS index     │       │  brain_mri.py        │
│  AraBERT embed   │       │  vision_base.py      │
└───────┬──────────┘       └──────────┬───────────┘
        │                             │
        ▼                             ▼
   Groq API / OpenAI           PyTorch + MONAI
   (LLaMA-3 / GPT-4o)         (EfficientNet / DenseNet / ResNet)
        │                             │
        └──────────────┬──────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   UTILS & INFRA                              │
│  utils/logger.py · utils/gradcam.py · utils/arabic_utils.py │
│  tests/ · train_*.py · core/config.py (.env)                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Principe de Lazy Loading (CRITIQUE)
Tous les modèles de vision sont chargés à la demande, jamais au démarrage.
Cela évite de saturer la RAM/VRAM dès le lancement de l'application.

```python
# PATTERN OBLIGATOIRE pour tous les modèles vision
@st.cache_resource  # Persiste entre les reruns Streamlit
def get_model(model_type: str):
    if model_type == "dermato":
        from engine.dermato import DermatoModel
        return DermatoModel()
    # etc.
```

### 3.3 Principe de Fallback Offline
Le système fonctionne à deux niveaux :
- **Niveau 1 (Online)** : RAG FAISS + Groq LLM → réponse riche et contextuelle
- **Niveau 2 (Offline)** : FAISS seul + formatage de réponse basique → réponse dégradée mais fonctionnelle

```python
# PATTERN OBLIGATOIRE dans llm.py
try:
    response = groq_client.chat(...)
except Exception:
    response = fallback_offline_response(context_from_faiss)
```

---

## ██ SECTION 4 — STRUCTURE DES FICHIERS (ÉTAT ACTUEL)

```
shifa-ai/
│
├── app.py                          ✅ OPÉRATIONNEL — Point d'entrée Streamlit
│
├── engine/                         # Cœur IA du système
│   ├── llm.py                      ✅ OPÉRATIONNEL — Interface Groq/OpenAI
│   ├── retriever.py                ✅ OPÉRATIONNEL — FAISS RAG engine
│   ├── prompt_builder.py           ✅ OPÉRATIONNEL — Construction prompts médicaux
│   ├── vision_base.py              🔧 EN COURS — Classe abstraite modèles vision
│   ├── vision_router.py            🔧 EN COURS — Auto-routing par type d'image
│   ├── dermato.py                  🔧 EN COURS — EfficientNet-B3 · 7 classes peau
│   ├── xray.py                     🔧 EN COURS — DenseNet-121 · Chest X-Ray
│   └── brain_mri.py                🔧 EN COURS — ResNet-50 + MONAI · IRM cérébrale
│
├── safety.py                       ✅ OPÉRATIONNEL — Filtrage sécurité médical
│
├── utils/
│   ├── logger.py                   ✅ OPÉRATIONNEL — Logging structuré
│   ├── gradcam.py                  ✅ OPÉRATIONNEL — Visualisation Grad-CAM
│   └── arabic_utils.py             ✅ OPÉRATIONNEL — Normalisation texte arabe
│
├── core/
│   └── config.py                   ✅ OPÉRATIONNEL — Settings + .env loader
│
├── tests/
│   └── test_pipeline.py            ✅ OPÉRATIONNEL — Tests unitaires pipeline RAG
│
├── train_dermato.py                📋 PLANIFIÉ — Fine-tuning EfficientNet-B3
├── train_xray.py                   📋 PLANIFIÉ — Fine-tuning DenseNet-121
├── train_arabert.py                📋 PLANIFIÉ — Fine-tuning AraBERT médical
│
├── .streamlit/
│   └── secrets.toml                🔐 LOCAL UNIQUEMENT — Jamais committé
│
├── requirements.txt                ✅ À MAINTENIR synchronisé
├── packages.txt                    ✅ Dépendances système (ffmpeg, libsndfile1)
└── .env.example                    ✅ Template variables d'environnement
```

---

## ██ SECTION 5 — MODULES IA EN DÉTAIL

### 5.1 Système RAG (Retrieval-Augmented Generation)

**Objectif** : Répondre aux questions médicales en arabe en s'appuyant sur une base
de connaissances locale, sans hallucination.

**Pipeline** :
```
Question arabe → AraBERT embedding → FAISS similarity search
→ Top-K documents récupérés → Injection dans prompt Groq
→ Réponse générée avec sources citées
```

**Principes à respecter** :
- Le contexte FAISS est toujours injecté AVANT la question dans le prompt
- Les sources sont toujours citées dans la réponse
- Le prompt système rappelle toujours que l'IA n'est pas un médecin
- Historique conversationnel limité aux 10 derniers échanges (mémoire glissante)
- Embeddings : AraBERT (768 dims), index FAISS IndexFlatL2

**Format prompt système obligatoire** :
```
أنت مساعد طبي ذكي يعمل ضمن نظام شفاء للذكاء الاصطناعي.
دورك هو مساعدة المرضى والأطباء في فهم الأعراض والحالات الطبية.
تذكر دائماً أنك لست بديلاً عن الطبيب المختص.
استخدم السياق التالي للإجابة: {context}
```

---

### 5.2 Modules Vision (En cours de finalisation)

#### Principes Communs à tous les modèles vision

1. **Héritage obligatoire** : Tout modèle vision hérite de `VisionBase`
2. **Prétraitement standardisé** :
   - Resize vers la taille optimale du modèle
   - Normalisation ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   - Conversion RGB obligatoire (les images médicales peuvent être RGBA ou grayscale)
3. **Inférence toujours sous `torch.no_grad()`** : pas de gradient en production
4. **Output standardisé** : chaque modèle retourne le même schema de dict
5. **Grad-CAM sur chaque prédiction** : toujours calculé, jamais optionnel
6. **Recommandation arabe** : chaque classe a sa recommandation en arabe

**Schema de sortie unifié (tous modèles vision)** :
```python
{
    "class": str,             # Classe prédite (nom complet)
    "confidence": float,      # 0.0 → 1.0
    "all_probs": dict,        # {classe: probabilité} pour toutes les classes
    "severity": str,          # "faible" | "modérée" | "élevée" | "critique"
    "urgency": str,           # "home_care" | "consult_doctor" | "emergency"
    "recommendation_ar": str, # Recommandation en arabe
    "gradcam": np.ndarray,    # Heatmap (H, W), valeurs 0.0–1.0
    "vision_type": str        # "dermato" | "xray" | "brain_mri"
}
```

#### Module Dermato (engine/dermato.py)
- **Architecture** : EfficientNet-B3 (torchvision, weights ImageNet pré-entraînés)
- **Taille entrée** : 300×300 px (optimum EfficientNet-B3)
- **Classes** (7 — HAM10000 dataset) :
  - Mélanocytaire bénin (nv) → Sévérité : faible
  - Mélanome (mel) → Sévérité : **critique**
  - Kératose bénigne (bkl) → Sévérité : faible
  - Carcinome basocellulaire (bcc) → Sévérité : **élevée**
  - Kératose actinique (akiec) → Sévérité : modérée
  - Lésion vasculaire (vasc) → Sévérité : modérée
  - Dermatofibrome (df) → Sévérité : faible
- **Grad-CAM target** : `model.features[-1]` (dernier bloc convolutif)
- **Head** : Dropout(0.3) + Linear(in_features → 7)

#### Module X-Ray (engine/xray.py)
- **Architecture** : DenseNet-121 (inspiré CheXNet — Stanford ML Group)
- **Taille entrée** : 224×224 px
- **Classes** (3) :
  - Normal → Urgence : home_care
  - Pneumonie bactérienne → Urgence : consult_doctor
  - Pneumonie virale / COVID-19 → Urgence : **emergency**
- **Grad-CAM target** : `model.features.denseblock4`
- **Head** : Linear(1024→256) + ReLU + Dropout(0.4) + Linear(256→3)

#### Module Brain MRI (engine/brain_mri.py)
- **Architecture** : ResNet-50 (torchvision) + MONAI preprocessing pipeline
- **Taille entrée** : 224×224 px
- **MONAI transforms** : ScaleIntensity → Resize → ToTensor
- **Classes** (4 — BraTS-inspired) :
  - Aucune tumeur → Sévérité : faible
  - Gliome → Sévérité : **critique**
  - Méningiome → Sévérité : **élevée**
  - Tumeur pituitaire → Sévérité : **élevée**
- **Grad-CAM target** : `model.layer4[-1]` (dernier bloc ResNet)
- **Note** : Les IRM peuvent être grayscale → convertir en 3 canaux par réplication

---

### 5.3 Safety Layer (safety.py)

**Principe fondamental** : TOUTE entrée passe par le safety layer AVANT traitement.
TOUTE sortie passe par le safety layer AVANT affichage.

**Règles de filtrage** :
1. Bloquer toute demande de prescription médicamenteuse directe
2. Bloquer toute demande d'auto-traitement dangereux
3. Détecter les signaux de détresse psychologique → rediriger vers urgences
4. Ajouter systématiquement le disclaimer médical en fin de réponse
5. Logger tous les inputs filtrés (sans données personnelles)

**Disclaimer obligatoire à ajouter à chaque réponse** :
```
⚠️ تنبيه مهم: هذه المعلومات للتوعية الصحية فقط ولا تغني عن استشارة الطبيب المختص.
```

---

### 5.4 Audio Pipeline

**Speech-to-Text** : Whisper (modèle `base` ou `small`) — local, offline-capable
- Langue principale : ar (arabe)
- Fallback : détection automatique de langue
- Format accepté : wav, mp3, m4a, ogg

**Text-to-Speech** : gTTS
- Langue : ar
- Vitesse : normale
- Lecture automatique de la réponse si mode audio activé

---

## ██ SECTION 6 — STANDARDS DE CODE OBLIGATOIRES

### 6.1 Structure de chaque fichier Python
```python
# ============================================================
# SHIFA AI · [Nom du module]
# Description : [1-2 lignes]
# Auteur : SHIFA AI Team
# ============================================================

import ... # stdlib d'abord
import ... # third-party ensuite
from ... import ... # imports internes en dernier

logger = logging.getLogger(__name__)

# Constantes en MAJUSCULES
CONSTANTE = valeur

class NomClasse:
    """Docstring obligatoire : rôle, inputs, outputs."""
    
    def methode(self, param: Type) -> Type:
        """Docstring courte."""
        pass
```

### 6.2 Typage
- Typage Python obligatoire sur TOUTES les fonctions
- Utiliser `from typing import Dict, List, Optional, Literal, Tuple`
- Les retours de fonctions IA toujours typés comme `Dict`

### 6.3 Gestion d'erreurs
```python
# PATTERN OBLIGATOIRE pour tous les appels IA
try:
    result = ai_call()
except SpecificException as e:
    logger.error(f"[ModuleName] Erreur spécifique: {e}")
    return fallback_response()
except Exception as e:
    logger.error(f"[ModuleName] Erreur inattendue: {e}")
    raise
```

### 6.4 Logging
- Utiliser `logger = logging.getLogger(__name__)` dans chaque module
- Niveaux : DEBUG pour dev, INFO pour flow normal, WARNING pour fallback, ERROR pour échecs
- Format log : `[NomModule] Message clair`
- NE JAMAIS logger de données médicales personnelles

### 6.5 Performance
- `@st.cache_resource` sur TOUS les chargements de modèles ML
- `@torch.no_grad()` sur TOUTES les fonctions d'inférence
- Lazy loading systématique : import du modèle dans la méthode `load_model()`, pas au niveau module
- Libérer la mémoire GPU après batch si modèle lourd : `torch.cuda.empty_cache()`

### 6.6 Internationalisation (i18n)
- Tous les messages utilisateur en ARABE
- Tous les messages de log et commentaires de code en FRANÇAIS ou ANGLAIS
- Utiliser `utils/arabic_utils.py` pour normaliser le texte arabe avant traitement
- Supporter RTL dans les composants Streamlit : `st.markdown('<div dir="rtl">...</div>', unsafe_allow_html=True)`

---

## ██ SECTION 7 — VARIABLES D'ENVIRONNEMENT

```bash
# .env.example — NE JAMAIS committer les vraies valeurs

# APIs IA
GROQ_API_KEY=gsk_xxxxxxxxxx              # LLM principal (LLaMA-3)
OPENAI_API_KEY=sk-xxxxxxxxxx             # Fallback LLM

# Configuration modèles
WHISPER_MODEL_SIZE=base                  # tiny | base | small | medium
FAISS_INDEX_PATH=./data/medical_index    # Chemin index vectoriel
EMBEDDING_MODEL=CAMeL-Lab/bert-base-arabic-camelbert-ca  # AraBERT

# Paramètres RAG
RAG_TOP_K=5                              # Nombre de documents récupérés
RAG_SIMILARITY_THRESHOLD=0.7            # Seuil similarité cosinus
CONVERSATION_HISTORY_SIZE=10            # Taille historique glissant

# Sécurité
SAFETY_FILTER_LEVEL=strict              # strict | moderate
LOG_LEVEL=INFO                          # DEBUG | INFO | WARNING | ERROR

# Device
FORCE_CPU=false                         # true = forcer CPU même si GPU dispo
```

---

## ██ SECTION 8 — STRATÉGIE DE DÉPLOIEMENT

### 8.1 Streamlit Cloud (Recommandé — Gratuit)
```
1. GitHub repo public → https://share.streamlit.io
2. Branch : main
3. Entry point : app.py
4. Secrets : copier contenu .env dans Settings → Secrets
5. Packages système : packages.txt (ffmpeg, libsndfile1)
```

**Optimisations obligatoires pour Streamlit Cloud** :
- `@st.cache_resource` sur tous les modèles (évite rechargement à chaque rerun)
- Modèles Whisper : utiliser `base` ou `small` (RAM limitée)
- FAISS index : committer dans le repo ou charger depuis HuggingFace Hub
- Modèles PyTorch : télécharger depuis `torchvision` au premier lancement (weights auto-download)

### 8.2 HuggingFace Spaces (Alternatif — GPU disponible)
```
1. New Space → SDK: Streamlit → Hardware: CPU basic (gratuit) ou T4 Small (GPU)
2. Uploader les fichiers ou connecter GitHub
3. Variables dans Settings → Repository secrets
4. README.md avec configuration HF Space header
```

**Header HF Space obligatoire (README.md)** :
```yaml
---
title: SHIFA AI
emoji: 🏥
colorFrom: green
colorTo: teal
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
---
```

### 8.3 Checklist Pré-Déploiement
- [ ] requirements.txt à jour avec versions pinned
- [ ] packages.txt présent (ffmpeg, libsndfile1)
- [ ] .env.example dans le repo (pas le .env réel)
- [ ] secrets.toml configuré sur la plateforme cible
- [ ] Tous les `@st.cache_resource` en place
- [ ] FAISS index présent ou URL de téléchargement définie
- [ ] Tests pipeline passent : `python -m pytest tests/`
- [ ] Safety layer actif et testé
- [ ] Disclaimer médical visible sur toutes les réponses

---

## ██ SECTION 9 — MODULES PLANIFIÉS (NE PAS IMPLÉMENTER SANS DEMANDE)

### 9.1 DDI Detector (Drug-Drug Interaction)
- Objectif : détecter les interactions médicamenteuses dangereuses
- Architecture prévue : GNN (Graph Neural Network) sur graphe d'interactions
- Dataset : DrugBank + données ONSSA Maroc
- Statut : **Non commencé** — attendre validation médicale

### 9.2 Fine-tuning AraBERT Médical
- Objectif : améliorer la compréhension des termes médicaux arabes dialectaux
- Dataset prévu : PubMed Arabic, corpus interne SHIFA
- Statut : **En préparation** — script train_arabert.py à compléter

### 9.3 Hiérarchisation des Rapports Médicaux
- Objectif : trier et prioriser les examens patients par urgence
- Statut : **Conception** uniquement

### 9.4 Calculateurs Médicaux Interactifs
- IMC, clairance créatinine, score de Glasgow, score de Wells DVT
- Statut : **Interface définie**, calculs à implémenter

---

## ██ SECTION 10 — CONTRAINTES ÉTHIQUES ET LÉGALES (NON NÉGOCIABLES)

1. **Jamais de diagnostic définitif** : Le système propose des hypothèses, jamais des certitudes
2. **Disclaimer systématique** : Chaque réponse se termine par un rappel de consulter un médecin
3. **Pas de prescription** : Le système ne prescrit jamais de médicaments ni de dosages
4. **Confidentialité** : Aucune donnée patient n'est loggée, stockée ou envoyée à des APIs tiers
5. **Transparence** : Le score de confiance est toujours affiché
6. **Limites affichées** : Si la confiance < 60%, le système affiche explicitement son incertitude
7. **Urgences** : Si urgence détectée, rediriger immédiatement vers le 15 (SAMU Maroc) ou urgences
8. **Pas de remplacement** : L'interface indique clairement à chaque session que SHIFA AI assiste, n'agit pas

---

## ██ SECTION 11 — QUESTIONS FRÉQUENTES POUR L'IA ASSISTANTE

**Q : Puis-je modifier la structure des dossiers ?**
R : Non, sauf demande explicite. La structure est définie et cohérente.

**Q : Puis-je changer de framework (ex: FastAPI au lieu de Streamlit) ?**
R : Non. Streamlit est le choix définitif pour ce projet.

**Q : Comment gérer un nouveau type d'image médicale ?**
R : Créer une nouvelle classe dans engine/ héritant de VisionBase, l'enregistrer dans vision_router.py.

**Q : Comment ajouter une nouvelle source de connaissances médicales ?**
R : Ajouter les documents dans le pipeline d'indexation FAISS, regénérer l'index.

**Q : Quelle est la priorité actuelle du projet ?**
R : (1) Finaliser les modules vision (dermato, xray, brain_mri), (2) Déployer sur Streamlit Cloud.

---

*Fin du SHIFA AI Master Prompt · À mettre à jour après chaque sprint majeur*
