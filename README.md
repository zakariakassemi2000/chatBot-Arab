# 🏥 شفاء AI — SHIFA AI

> Plateforme médicale intelligente en langue arabe
> Diagnostic · Analyse d'images · RAG médical · Audio

![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C)
![Groq](https://img.shields.io/badge/Groq-LLaMA3_70b-F55036)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Fonctionnalités

| Module | Description | Statut |
|--------|-------------|--------|
| 🤖 محادثة طبية | Chat médical arabe avec RAG FAISS 7975 docs | ✅ Live |
| 🔬 تحليل الصور | Vision IA : Peau · Poumons · Cerveau · Cancer · Sein | ✅ Live |
| 🎙️ المساعد الصوتي | Whisper ASR + gTTS arabe | ✅ Live |
| 🛡️ Safety Layer | Filtrage médical + disclaimer automatique | ✅ Live |

## 🏗️ Stack Technique

- **Frontend** : Streamlit + CSS Dark Medical Premium
- **LLM** : Groq API (LLaMA-3 70b) + fallback offline
- **Vision** : PyTorch + MONAI (EfficientNet / DenseNet / ResNet)
- **RAG** : FAISS + CAMeL-Lab AraBERT (7975 documents)
- **Audio** : OpenAI Whisper + gTTS

## 🚀 Installation locale

```bash
git clone https://github.com/zakariakassemi2000/chatBot-Arab
cd chatBot-Arab
pip install -r requirements.txt
```

Crée un fichier `.env` :
```text
GROQ_API_KEY=ta_cle_ici
```

Lance l'application :
```bash
streamlit run app.py
```

## ⚠️ Avertissement Médical

SHIFA AI est un outil d'assistance uniquement.
Il ne remplace pas un médecin qualifié.
Consultez toujours un professionnel de santé.
