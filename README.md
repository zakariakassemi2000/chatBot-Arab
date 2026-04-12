# 🧠 SHIFA-Mental — Module de Soutien Psychologique

## Architecture Globale

```
╔══════════════════════════════════════════════════════════════╗
║               SHIFA-Mental (شفاء-نفس)                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ┌─────────────────────────────────────────────────────┐    ║
║  │           Interface Streamlit RTL Arabe              │    ║
║  │   💬 Chat  │  📋 PHQ-9  │  🫁 Relaxation  │  📍 Resources │
║  └──────────────────────┬──────────────────────────────┘    ║
║                         │                                    ║
║  ┌──────────────────────▼──────────────────────────────┐    ║
║  │              Distress Detection Engine               │    ║
║  │                                                      │    ║
║  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │    ║
║  │  │  Keyword NLP │  │  PHQ-9 Score │  │  LLM Signal│ │    ║
║  │  │  (Arabic)   │  │  Calculator  │  │  [CRISIS]  │ │    ║
║  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │    ║
║  │         └───────────┬────┘                │        │    ║
║  │                     ▼                     │        │    ║
║  │           Severity Level (0-3)             │        │    ║
║  └─────────────────────┬─────────────────────┘        │    ║
║                        │                              │     ║
║     ┌──────────────────▼──────────────────────────────┘     ║
║     │                                                        ║
║     │  Level 0 → Normal Response                            ║
║     │  Level 1 → Supportive + Relaxation Suggestion         ║
║     │  Level 2 → CBT Techniques + Professional Referral     ║
║     │  Level 3 → CRISIS ALERT → Hotline Routing (فوري)     ║
║     │                                                        ║
║  ┌──▼──────────────────────────────────────────────────┐    ║
║  │              LLM Psychotherapy Engine                │    ║
║  │                                                      │    ║
║  │  System Prompt: CBT Therapist + Islamic Counseling  │    ║
║  │                                                      │    ║
║  │  Option A: Jais-13B (Fine-tuned) ← Production      │    ║
║  │  Option B: Claude API + Psychotherapy Prompt        │    ║
║  │  Option C: HF Inference API                        │    ║
║  └──────────────────────────────────────────────────────┘   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Structure des Fichiers

```
shifa_mental/
├── mental_module.py          # Module principal Streamlit
├── llm_finetuning_guide.py   # Guide fine-tuning Jais-13B
├── README.md                 # Ce fichier
└── datasets/                 # (à créer)
    ├── arabic_psychotherapy_synthetic.jsonl
    ├── phq9_validation_arabic.json
    └── crisis_keywords_arabic.json
```

## Intégration dans SHIFA AI Principal

Dans `app.py` ou `main.py` SHIFA :

```python
from shifa_mental.mental_module import render_mental_module

# Dans le sidebar/navigation SHIFA :
if selected_module == "الصحة النفسية":
    render_mental_module(api_key=st.secrets.get("ANTHROPIC_API_KEY"))
```

## Pipeline de Détection de Détresse

```
Texte Utilisateur (Arabe)
         │
         ▼
┌─────────────────────────┐
│   Keyword Matching       │
│   (Crisis/High/Mid)      │
└──────────┬──────────────┘
           │
     ┌─────▼──────┐
     │  Level 3?  │──YES──▶ Alerte Crise immédiate
     └─────┬──────┘         + Routage SAMU Social
           │NO
     ┌─────▼──────┐
     │  Level 2?  │──YES──▶ CBT + Référence Psy
     └─────┬──────┘
           │NO
     ┌─────▼──────┐
     │  Level 1?  │──YES──▶ Soutien + Exercices
     └─────┬──────┘
           │NO
           ▼
    Réponse Générale
    (Prévention/Bien-être)
```

## Fine-Tuning Jais-13B — Étapes

| Étape | Tâche | Durée |
|-------|-------|-------|
| 1 | Collecte données (Twitter AR + synthetic) | 1 semaine |
| 2 | Nettoyage + annotation severité | 3 jours |
| 3 | Format ChatML + split train/test | 1 jour |
| 4 | QLoRA training sur A100 (Google Colab Pro+) | 6 heures |
| 5 | Évaluation clinique + safety tests | 2 jours |
| 6 | Déploiement HF Inference API | 1 jour |

**Coût estimé** : ~20-30$ (Google Colab Pro+ A100)

## Variables Streamlit Secrets

```toml
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "sk-ant-..."
HF_TOKEN = "hf_..."           # Pour Jais-13B sur HF
MENTAL_MODULE_ENABLED = "true"
```

## Sécurité & Éthique

- ⚠️ **Disclaimer** affiché à chaque session : "outil de soutien, non substitut thérapeutique"
- 🔒 **Données** : aucune conversation stockée (session state only)
- 🚨 **Crisis Override** : si `[CRISIS_DETECTED]` → routing immédiat, LLM ne continue pas
- 📋 **PHQ-9** : validé cliniquement, scores interprétés selon seuils Kroenke & Spitzer (2002)
- 🌐 **Ressources** : numéros marocains vérifiés (0800 005 100, SAMU Social)

## Roadmap SHIFA-Mental v2

- [ ] Analyse vocale de détresse (Whisper → tonalité + disfluences)
- [ ] Journalisation émotionnelle (mood tracker persistant)
- [ ] Exercices EMDR simplifiés (stimulation bilatérale visuelle)
- [ ] Intégration DSM-5 arabe pour screening automatique
- [ ] Partenariat avec psychologues marocains pour supervision
