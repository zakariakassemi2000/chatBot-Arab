# 🏥 SHIFA AI - المنصة الطبية الذكية (Version 2.0)

![SHIFA AI Banner](https://img.shields.io/badge/SHIFA-AI_Medical_Assistant-E53935?style=for-the-badge&logo=health&logoColor=white) ![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?style=flat-square&logo=streamlit)

SHIFA AI est un assistant médical avancé propulsé par l'IA (Vision, LLM, RAG) conçu pour fournir des diagnostics préliminaires, l'analyse d'images médicales, la recherche de médecins à proximité, et la vérification d'interactions médicamenteuses.

---

## 📑 Sommaire
1. [README Amélioré](#1-readme-amélioré)
2. [دليل المستخدم (Guide Utilisateur en Arabe)](#2-دليل-المستخدم-guide-utilisateur)
3. [Guide Développeur (Architecture & API)](#3-guide-développeur)
4. [Changelog V2.0](#4-changelog-v20)

---

## 1. README Amélioré

### ✨ Nouvelles Fonctionnalités (V2.0)
SHIFA AI a été entièrement remanié pour offrir une expérience utilisateur (UX) de niveau "Production" :

* **Tableau de Bord Centralisé (Grid System)** : Disparition de l'ancienne barre latérale encombrée au profit d'une grille de 12 cartes interactives permettant d'accéder à tous les modules en un clic.
* **Bannière d'Urgences Intégrée** : Affichage fixe des numéros d'urgence vitaux (Ambulance 15, Police 19) avec animations (`pulse`) pour dissuader l'usage de l'IA en cas de crise immédiate.
* **Statistiques Système Temps Réel** : Monitoring en direct de l'état des moteurs IA, de la base de données vectorielle (FAISS) et des temps de latence.
* **Système de Géolocalisation Natif (Overpass API)** : Recherche de soins à proximité totalement gratuite et open-source, en remplacement complet de Google Maps.
* **Routeur de Vision Unifié** : Un moteur unique traitant de manière transparente la Dermatologie, les Rayons-X, l'IRM cérébrale et l'Oncologie avec des modèles spécialisés (EfficientNet-B3, ViT, etc.).

### 📸 Captures d'Écran
> **Note :** Les captures peuvent être ajoutées dans le dossier `/docs/images/`
* `dashboard_v2.png` : **Tableau de bord dynamique** affichant les cartes interactives et les voyants d'état verts/rouges.
* `vision_unified.png` : **Interface de diagnostic**, montrant la détection de la classe médicale (ex: Mélanome) avec la carte de chaleur dynamique (Grad-CAM).
* `nearby_care_osm.png` : **Recherche par GPS** avec carte Folium interactive et liste des cliniques/hôpitaux à proximité.

### ⚙️ Guide d'Installation

1. **Prérequis Système** : Python 3.10+ et Git.
   *(Sur Linux/Streamlit Cloud, installez `libglib2.0-0` et `libgomp1` pour OpenCV et PyTorch).*

2. **Clonage et Dépendances** :
   ```bash
   git clone https://github.com/votre-compte/shifa-ai.git
   cd shifa-ai
   python -m venv venv
   source venv/bin/activate  # Sous Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Variables d'Environnement** :
   Créez un fichier `.env` ou `.streamlit/secrets.toml` :
   ```toml
   GROQ_API_KEY = "gsk_..."
   HF_TOKEN = "hf_..."
   APP_PASSWORD = "votre_mot_de_passe"
   ```

4. **Lancement** :
   ```bash
   streamlit run app.py
   ```

---

## 2. دليل المستخدم (Guide Utilisateur)

مرحباً بك في **SHIFA AI** (شفاء)، مساعدك الطبي الذكي. يقدم لك هذا الدليل شرحاً مبسطاً لكيفية الاستفادة القصوى من ميزات التطبيق.

### 🔴 حالات الطوارئ (اللافتة الحمراء)
فور دخولك للتطبيق، ستلاحظ لافتة حمراء وامضة في الأعلى إذا كنت تواجه حالة طوارئ حقيقية (مثل ألم شديد في الصدر أو صعوبة في التنفس)، **يرجى عدم استخدام التطبيق**. استخدم الأرقام الموضحة فوراً (🚑 15 أو 150، 🚓 19).

### 🤖 كيفية استخدام المحادثة الطبية (CTA الرئيسي)
1. انقر على بطاقة **"🤖 محادثة طبية"** في الشاشة الرئيسية.
2. يمكنك اختيار أحد "الأسئلة السريعة" الجاهزة، أو كتابة أعراضك بالتفصيل في صندوق الدردشة السفلية.
3. سيرد عليك محرك `Groq` بمعلومات طبية دقيقة مستمدة من قواعد البيانات. للعودة للرئيسية، اضغط على زر "🏠 الرجوع للرئيسية" في القائمة الجانبية.

### 🔬 كيفية تحليل الصور الطبية
1. اختر **"🔬 تحليل الصور"** من الشاشة الرئيسية.
2. قم برفع الصورة الطبية التي تمتلكها (أشعة سينية، رنين مغناطيسي لدماغ، أو صورة لمرض جلدي).
3. سيفحص النظام الصورة تلقائياً ويتعرف على نوعها، ثم يجري التحليل المتقدم للبحث عن أي تشوهات أو أمراض محتملة، مع عرض نسبة الثقة (أرقام مئوية) وتظليل المناطق المصابة.

### 🏥 كيفية العثور على طبيب قريب
1. انقر على **"🏥 الرعاية القريبة"**.
2. اسمح للمتصفح بتحديد موقعك الجغرافي، وسيقوم النظام فوراً بالبحث في خريطة مفتوحة المصدر عن أقرب العيادات والمستشفيات إليك (في دائرة 5-10 كيلومتر).
3. ستظهر بطاقات بأسماء الأطباء وتخصصاتهم والمسافة التي تفصلك عنهم.

### ⚠️ كيفية قراءة التوصيات
ستُعرض نتائج التطبيق بألوان واضحة لتحديد مستوى الخطورة:
* **الأخضر (روتيني/بسيط):** حالة لا تستدعي القلق السريع.
* **الأصفر (متوسط/استشارة طبيب):** يُنصح بأخذ موعد طبي في أقرب فرصة.
* **الأحمر (طارئ):** يتطلب تدخلاً طبياً عاجلاً وفورياً.
*(تذكر أن SHIFA AI لا يعوض زيارة الطبيب وتشخيصه النهائي).*

---

## 3. Guide Développeur

### 🏗️ Architecture des Nouvelles Pages (App.py vs Pages/)
Le projet fonctionne désormais avec une **architecture Single Page Application (SPA) hybride** :
- `app.py` contient l'orchestrateur de la vue et gère sa propre navigation simulée pour les modules fondamentaux via `st.session_state.page` (Chat, Vision, Voix).
- Les **fichiers multlipages isolés** (ex: `pages/10_🏥_الرعاية_القريبة.py`) coexistent mais la barre latérale standard de Streamlit (`[data-testid="stSidebarNav"]`) est **masquée** via CSS pour empêcher la répétition de navigation.
- Le passage entre l'orchestrateur principal et les pages autonomes se fait via les boutons du dashboard utilisant `st.switch_page("pages/fichier.py")`.

### 🔌 API des Fonctions Helper 
Le projet met en œuvre des helpers centraux :
- **`utils/geolocation.py`** : Fallback double (HTML5 Geolocation API → IP-API JSON). Gère les timeouts et les rejets de permission avec des `st.session_state` de secours.
- **`engine/safety.py`** : Intercepteur agissant comme un "Guardrail". Contient `detect_emergency(text)` qui utilise la distance de Levenshtein avec un dictionnaire de mots-clés d'urgence arabe.
- **`utils/image_validator.py`** : La méthode `validate_medical_image` prévient le téléchargement d'images "poubelles" en vérifiant via `OpenAI CLIP` (Zero-Shot) si l'image téléchargée est réellement une image médicale.

### 🗺️ Intégration Overpass API (OpenStreetMap)
Remplacement complet de Google Maps :
1. **Requêtes Nominatim** : La fonction `get_nearby_hospitals` effectue une requête HTTP asynchrone sur l'API Overpass (`overpass-api.de/api/interpreter`) en injectant les coordonnées de la `BBox` (Bounding Box).
2. **Traductions** : Utilisation de tags OSM natifs (`amenity=hospital`, `amenity=doctors`) mappés sur des étiquettes en arabe (`AMENITY_LABELS`).
3. **Cartographie Folium** : Rendu de la carte interactive utilisant les bibliothèques `folium` et `streamlit-folium` (carte hors ligne OpenStreetMap standard).

### 🎨 Personnalisation CSS
Le nouveau design est piloté par un bloc `<style>` directement injecté par une fonction Python globale :
* **Gradients interactifs** : `.landing-card` expose un fond `rgba` transparent (Glassmorphism) et a un pseudo-élément `:before` configuré avec une déformation `skewX(-25deg)` animée au `:hover` pour de la brillance.
* **Typographie** : Importation de l'arabe *Cairo* de Google Fonts pour toute l'interface `[class*="st-"] { font-family: 'Cairo'; }`

---

## 4. Changelog (Version 2.0)
**Version 2.0.0 "Production Readiness Update"** — Mars 2026

### 🚀 Améliorations UI/UX (Les 10 Features Ajoutées)
1. **[UI] Nouveau Dashboard :** Menu par grille cartographique (12 cartes) remplaçant le panneau latéral Streamlit par défaut.
2. **[UI] Bannière Urgence (Rouge) :** Disclaimers de secours affichés en permanence de façon responsive.
3. **[UX] Statistiques Vives :** Indicateurs dynamiques du statut AI/DB.
4. **[UX] Fast-Resume (Historique) :** Section expander pour reprendre instantanément la conversation en cours depuis l'accueil.
5. **[Architecture] Nettoyage des Pages :** Suppression des instances dupliquées (Cerveau, Poumons, Dermatologie) vers le routeur `VisionRouter` unifié de `app.py`.
6. **[Maps] Open Source Geo :** Abandon de Google Maps (payants) au profit d'OpenStreetMap/Folium, sans besoin de CB.
7. **[Forms] Login Gate sécurisé :** Implémentation d'une page de mot de passe stricte (APP_PASSWORD).
8. **[UI] Animations Avancées :** Introduction du CSS `@keyframes` (pulse, slide, hover transform) pour les Alertes et Cartes.
9. **[Safety] Fix `sys.exit()` :** Remplacement des commandes fatales causant un plantage sur le Cloud par `st.stop()`.
10. **[Media] Corrections Voix :** Ajustement asynchrone des modèles STT avec un context manager `tempfile.NamedTemporaryFile` sécurisé.

### ⚠️ Breaking Changes
- Le dossier `pages/` ne supporte plus les vues isolées `01_` à `05_`. L'accès au Chat et à la Vision se fait de manière incontournable via la page centrale (`app.py`).
- Fin du support de l'API Google Maps et des clés associées.
- Le projet requiert impérativement Streamlit `1.42+` pour exploiter correctement `st.switch_page()` et la navigation interne unifiée sans la barre native.
