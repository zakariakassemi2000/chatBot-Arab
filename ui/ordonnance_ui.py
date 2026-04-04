# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — UI Scanner d'Ordonnance
  Composants Streamlit : upload, webcam, cartes médicales, disclaimers
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import logging

logger = logging.getLogger("shifa.ordonnance_ui")


def _inject_ordonnance_css():
    """Injecte les styles CSS spécifiques au module ordonnance."""
    st.markdown("""
    <style>
        /* ═══ Animation d'entrée ═══ */
        @keyframes ordFadeIn {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .ord-animate {
            animation: ordFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        /* ═══ Carte Médicament ═══ */
        .med-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
            border: 1px solid rgba(22, 163, 74, 0.2);
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s, box-shadow 0.3s;
            animation: ordFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .med-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(22, 163, 74, 0.2);
        }

        /* Accent bar on the right */
        .med-card::before {
            content: "";
            position: absolute;
            top: 0; right: 0;
            width: 5px; height: 100%;
            border-radius: 0 16px 16px 0;
        }
        .med-card.high::before   { background: linear-gradient(to bottom, #16a34a, #22d3ee); }
        .med-card.medium::before { background: linear-gradient(to bottom, #d4af37, #f97316); }
        .med-card.low::before    { background: linear-gradient(to bottom, #dc2626, #f87171); }

        /* Card header */
        .med-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        .med-name {
            font-size: 1.35rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0;
            font-family: 'Cairo', sans-serif;
        }
        .med-principe {
            font-size: 0.9rem;
            color: #94a3b8;
            margin: 2px 0 0 0;
            font-style: italic;
        }

        /* Badge */
        .med-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .med-badge.found {
            background: rgba(22, 163, 74, 0.15);
            color: #4ade80;
            border: 1px solid rgba(22, 163, 74, 0.3);
        }
        .med-badge.notfound {
            background: rgba(220, 38, 38, 0.1);
            color: #fca5a5;
            border: 1px solid rgba(220, 38, 38, 0.2);
        }

        /* Info grid */
        .med-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.8rem;
            margin: 1rem 0;
        }
        .med-info-item {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 10px;
            padding: 0.65rem 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }
        .med-info-label {
            font-size: 0.72rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 3px;
        }
        .med-info-value {
            font-size: 0.95rem;
            color: #e2e8f0;
            font-weight: 600;
        }

        /* Confidence bar */
        .conf-bar-bg {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin-top: 0.8rem;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .conf-bar-fill.high   { background: linear-gradient(90deg, #16a34a, #22d3ee); }
        .conf-bar-fill.medium { background: linear-gradient(90deg, #d4af37, #f97316); }
        .conf-bar-fill.low    { background: linear-gradient(90deg, #dc2626, #f87171); }

        .conf-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Global score card */
        .score-global {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.1), rgba(34, 211, 238, 0.08));
            border: 1px solid rgba(22, 163, 74, 0.25);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .score-number {
            font-size: 3rem;
            font-weight: 800;
            margin: 0.5rem 0;
        }
        .score-number.high   { color: #4ade80; }
        .score-number.medium { color: #fbbf24; }
        .score-number.low    { color: #f87171; }

        /* Disclaimer */
        .ord-disclaimer {
            background: rgba(220, 38, 38, 0.06);
            border-right: 4px solid #dc2626;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-top: 2rem;
        }
        .ord-disclaimer p {
            color: #fca5a5;
            font-size: 0.85rem;
            line-height: 1.7;
            margin: 0;
        }
        .ord-disclaimer b {
            color: #f87171;
        }

        /* OCR text display */
        .ocr-text-box {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1.2rem;
            font-family: 'Courier New', monospace;
            font-size: 0.88rem;
            color: #94a3b8;
            white-space: pre-wrap;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }

        /* Formes pills */
        .forme-pill {
            display: inline-block;
            background: rgba(212, 175, 55, 0.12);
            color: #d4af37;
            border: 1px solid rgba(212, 175, 55, 0.25);
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 2px 3px;
        }
    </style>
    """, unsafe_allow_html=True)


def _get_confidence_class(score: float) -> str:
    """Retourne la classe CSS en fonction du score."""
    if score >= 75:
        return "high"
    elif score >= 50:
        return "medium"
    return "low"


def _render_medication_card(med, index: int) -> str:
    """Génère le HTML d'une carte médicale stylisée. Retourne le HTML (ne fait PAS de st.markdown)."""
    conf_class = _get_confidence_class(med.score_match)

    # Badge status
    if med.est_reference:
        badge = '<span class="med-badge found">✓ Base marocaine</span>'
    else:
        badge = '<span class="med-badge notfound">✗ Non référencé</span>'

    # Formes disponibles pills
    formes_html = ""
    if med.formes_disponibles:
        pills = "".join(f'<span class="forme-pill">{f}</span>' for f in med.formes_disponibles)
        formes_html = f'<div style="margin-top:0.5rem;">{pills}</div>'

    # Nom affiché
    display_name = med.nom_match if med.nom_match else med.nom_brut
    principe_html = f'<p class="med-principe">🧬 {med.principe_actif}</p>' if med.principe_actif else ""

    # Info items
    info_items = []
    if med.dosage:
        info_items.append(("💊 Dosage", med.dosage))
    if med.forme:
        info_items.append(("📦 Forme", med.forme.capitalize()))
    if med.frequence:
        info_items.append(("🔄 Fréquence", med.frequence))
    if med.duree:
        info_items.append(("📅 Durée", med.duree))

    info_grid = ""
    if info_items:
        items_html = "".join(
            f'<div class="med-info-item">'
            f'<div class="med-info-label">{label}</div>'
            f'<div class="med-info-value">{value}</div>'
            f'</div>'
            for label, value in info_items
        )
        info_grid = f'<div class="med-info-grid">{items_html}</div>'

    # Confidence bar (compact — no line breaks)
    score_pct = med.score_match if med.est_reference else 0
    conf_bar = ""
    if med.est_reference:
        color = '#4ade80' if conf_class == 'high' else '#fbbf24' if conf_class == 'medium' else '#f87171'
        conf_bar = (
            f'<div class="conf-bar-bg">'
            f'<div class="conf-bar-fill {conf_class}" style="width:{score_pct}%"></div>'
            f'</div>'
            f'<div class="conf-label">'
            f'<span>Confiance du matching</span>'
            f'<span style="color:{color};font-weight:700">{score_pct:.0f}%</span>'
            f'</div>'
        )

    # Build card HTML — all on connected lines (no blank lines that break Streamlit HTML)
    card_html = (
        f'<div class="med-card {conf_class}" style="animation-delay:{index * 0.1}s">'
        f'<div class="med-card-header">'
        f'<div><h3 class="med-name">💊 {display_name}</h3>{principe_html}</div>'
        f'{badge}'
        f'</div>'
        f'{info_grid}'
        f'{formes_html}'
        f'{conf_bar}'
        f'</div>'
    )
    return card_html


def render_ordonnance_page():
    """
    Page Streamlit complète pour le scanner d'ordonnance.
    Inclut : upload/webcam, analyse OCR, cartes médicales, disclaimers.
    """

    _inject_ordonnance_css()

    # ── Titre ──
    st.markdown(
        '<div class="moroccan-title" style="font-size:2.8rem;">📋 ماسح الوصفات الطبية</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="moroccan-subtitle">Scanner une ordonnance · استخراج الأدوية تلقائيا بالذكاء الاصطناعي</div>',
        unsafe_allow_html=True
    )

    # ── Disclaimer haut ──
    st.markdown("""
    <div class="ord-disclaimer" style="margin-bottom:1.5rem; border-right-color: #d4af37;">
        <p>
            <b>⚕️ Avertissement important :</b><br/>
            Ce scanner utilise la reconnaissance optique de caractères (OCR) pour extraire les informations
            d'une ordonnance. Les résultats sont <b>indicatifs</b> et ne remplacent en aucun cas
            la lecture professionnelle par un pharmacien ou un médecin.
            <b>Consultez toujours votre professionnel de santé.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Source d'image ──
    with st.container(border=True):
        st.markdown(
            "<h4 style='color:#d4af37; margin-bottom:1rem;'>📷 Source de l'ordonnance</h4>",
            unsafe_allow_html=True
        )

        tab_upload, tab_camera = st.tabs(["📁 Importer un fichier", "📸 Webcam"])

        image = None

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Glissez une photo d'ordonnance ici",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Formats acceptés : JPG, PNG, BMP, TIFF",
                key="ord_upload"
            )
            if uploaded_file:
                from PIL import Image as PILImage
                image = PILImage.open(uploaded_file)

        with tab_camera:
            camera_photo = st.camera_input(
                "Prenez une photo de l'ordonnance",
                key="ord_camera"
            )
            if camera_photo:
                from PIL import Image as PILImage
                image = PILImage.open(camera_photo)

    # ── Traitement ──
    if image is not None:
        with st.container(border=True):
            col_img, col_action = st.columns([1.2, 1])

            with col_img:
                st.image(image, caption="Ordonnance chargée", use_container_width=True)

            with col_action:
                st.markdown("""
                <div style="padding: 1rem 0;">
                    <h4 style="color: #f8fafc; margin-bottom: 0.5rem;">🔍 Analyse OCR</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        Le système va :<br/>
                        ✓ Pré-traiter l'image (contraste, netteté)<br/>
                        ✓ Extraire le texte en français (Tesseract)<br/>
                        ✓ Identifier les médicaments et posologies<br/>
                        ✓ Matcher avec la base pharmaceutique marocaine
                    </p>
                </div>
                """, unsafe_allow_html=True)

                analyze_btn = st.button(
                    "🚀 Lancer l'analyse",
                    type="primary",
                    use_container_width=True,
                    key="ord_analyze"
                )

        if analyze_btn:
            with st.spinner("🔬 Analyse OCR en cours... Pré-traitement → Extraction → Matching"):
                try:
                    from modules.ordonnance_scanner import analyze_ordonnance
                    result = analyze_ordonnance(image)
                except ImportError as ie:
                    st.error(f"⚠️ Module manquant : {ie}")
                    st.info("Installez les dépendances : `pip install pytesseract rapidfuzz`")
                    return
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")
                    return

            # ── Erreur ? ──
            if result.erreur:
                st.error(f"⚠️ {result.erreur}")
                if result.texte_brut:
                    with st.expander("📝 Texte OCR brut extrait"):
                        st.markdown(
                            f'<div class="ocr-text-box">{result.texte_brut}</div>',
                            unsafe_allow_html=True
                        )
                return

            # ── Score global ──
            score_class = _get_confidence_class(result.score_global)
            st.markdown(f"""
            <div class="score-global ord-animate">
                <div style="color: #94a3b8; font-size: 0.9rem; font-weight: 600;">
                    SCORE DE CONFIANCE GLOBAL
                </div>
                <div class="score-number {score_class}">
                    {result.score_global:.0f}%
                </div>
                <div style="color: #64748b; font-size: 0.82rem;">
                    {len(result.medicaments)} médicament(s) détecté(s)
                    {'· ' + str(sum(1 for m in result.medicaments if m.est_reference)) + ' référencé(s) dans la base marocaine' if result.medicaments else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Cartes médicaments ──
            if result.medicaments:
                # Build ALL cards HTML in one block to prevent Streamlit
                # from breaking the HTML context between separate st.markdown calls
                all_cards_html = "<h3 style='color:#d4af37;margin:1.5rem 0 1rem'>💊 Médicaments détectés</h3>"
                for i, med in enumerate(result.medicaments):
                    all_cards_html += _render_medication_card(med, i)
                st.markdown(all_cards_html, unsafe_allow_html=True)

            else:
                st.warning(
                    "Aucun médicament n'a pu être identifié dans cette ordonnance. "
                    "Essayez avec une image plus nette ou un meilleur éclairage."
                )

            # ── Texte OCR brut ──
            with st.expander("📝 Texte OCR brut extrait (debug)"):
                st.markdown(
                    f'<div class="ocr-text-box">{result.texte_brut}</div>',
                    unsafe_allow_html=True
                )

            # ── Disclaimer final ──
            st.markdown("""
            <div class="ord-disclaimer">
                <p>
                    <b>⚠️ AVERTISSEMENT LÉGAL — MENTIONS OBLIGATOIRES :</b><br/><br/>
                    ① Les résultats de ce scanner sont générés par un système automatisé de reconnaissance
                    optique de caractères (OCR) et de matching algorithmique. Ils sont fournis
                    <b>à titre informatif uniquement</b>.<br/><br/>
                    ② Ce système <b>ne constitue pas un avis médical ni pharmaceutique</b>.
                    Les informations extraites peuvent contenir des erreurs dues à la qualité
                    de l'image, à l'écriture manuscrite ou aux limites de la technologie OCR.<br/><br/>
                    ③ <b>Consultez systématiquement votre médecin ou votre pharmacien</b>
                    pour la validation de toute ordonnance avant la prise de médicaments.<br/><br/>
                    ④ SHIFA AI décline toute responsabilité en cas d'utilisation inappropriée
                    des informations fournies par ce module.<br/><br/>
                    <span style="color:#64748b; font-size:0.78rem;">
                        Conformément à la réglementation marocaine (Loi 17-04 portant code du médicament
                        et de la pharmacie), seul un professionnel de santé habilité peut prescrire
                        et délivrer des médicaments.
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
