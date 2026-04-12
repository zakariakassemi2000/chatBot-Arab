# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — UI Scanner d'Ordonnance
  Composants Streamlit : upload, webcam, cartes médicales, disclaimers
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import logging
import io

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
        '<div class="moroccan-subtitle">Scanner une ordonnance · استخراج الأدوية تلقائيا</div>',
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
                st.image(image, caption="Ordonnance chargée", width='stretch')

            with col_action:
                st.markdown("""
                <div style="padding: 1rem 0;">
                    <h4 style="color: #f8fafc; margin-bottom: 0.5rem;">🔍 Analyse OCR</h4>
                    <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
                        Le système va :<br/>
                        ✓ Pré-traiter l'image (contraste, netteté)<br/>
                        ✓ Extraire le texte de l'ordonnance<br/>
                        ✓ Identifier les médicaments et posologies<br/>
                        ✓ Matcher avec la base pharmaceutique marocaine<br/>
                        ✓ Vérifier le prix sur medicament.ma et le remboursement CNOPS
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                analyze_btn = st.button(
                    "🚀 Lancer l'analyse",
                    type="primary",
                    width='stretch',
                    key="ord_analyze"
                )

        if analyze_btn:
            with st.spinner("🔬 Analyse en cours..."):
                try:
                    from engine.vision_ocr.vlm_extraction import extract_from_image
                    
                    # Convert PIL Image to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format or 'JPEG')
                    image_bytes = img_byte_arr.getvalue()

                    extraction = extract_from_image(image_bytes)
                    st.session_state["ocr_extraction"] = extraction
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")
                    logger.error(f"OCR error: {e}", exc_info=True)
                    return

        # ── Affichage et Correction interactive ──
        if st.session_state.get("ocr_extraction") is not None:
            extraction = st.session_state["ocr_extraction"]
            import engine.vision_ocr.decision_engine as dec_engine
            
            score_global = extraction.confiance_globale * 100
            score_class = _get_confidence_class(score_global)
            n_meds = len(extraction.medicaments) if hasattr(extraction, 'medicaments') and extraction.medicaments else 0

            st.markdown(f"""
            <div class="score-global ord-animate">
                <div style="color: #94a3b8; font-size: 0.9rem; font-weight: 600;">
                    INDICE DE CONFIANCE
                </div>
                <div class="score-number {score_class}">
                    {score_global:.0f}%
                </div>
                <div style="color: #64748b; font-size: 0.82rem;">
                    {n_meds} médicament(s) détecté(s)
                </div>
            </div>
            """, unsafe_allow_html=True)

            if n_meds > 0:
                st.markdown("<h3 style='color:#d4af37;margin:1.5rem 0 1rem'>💊 Médicaments détectés</h3>", unsafe_allow_html=True)
                for i, med in enumerate(extraction.medicaments):
                    analysis = dec_engine.analyze_medication(
                        raw_name=med.nom,
                        raw_dosage=med.dosage,
                        vlm_confidence=extraction.confiance_globale,
                        posologie=med.posologie,
                        duree=med.duree,
                    )
                    
                    status = analysis["status"]
                    conf_pct = analysis["confidence"] * 100
                    conf_class = _get_confidence_class(conf_pct)
                    
                    if status == "valid":
                        badge = '<span class="med-badge found">✓ Validé</span>'
                    elif status == "suspect":
                        badge = '<span class="med-badge" style="background:rgba(234,179,8,0.15);color:#eab308;border:1px solid rgba(234,179,8,0.3)">⚠️ Suspect</span>'
                    else:
                        badge = '<span class="med-badge notfound">✗ Non trouvé</span>'

                    display_name = analysis.get("corrected_name", med.nom)
                    
                    status_text = "Valid" if status == "valid" else "Suspect" if status == "suspect" else "Non trouvé"
                    val_dosage = analysis.get('dosage') or "Non détecté"
                    val_poso = analysis.get('posologie') or "Non détectée"
                    val_duree = analysis.get('duree') or "Non détectée"
                    val_prix = f"{analysis['price']} DH" if analysis.get('price') else "Non trouvé"
                    val_cnops = "Oui ✅" if analysis.get("remboursable") else "Non 🚫"
                    val_type = analysis.get("type") or "Unknown"

                    list_html = f"""
                    <div style="background: rgba(15, 23, 42, 0.4); border-radius: 8px; padding: 1rem; border: 1px solid rgba(255, 255, 255, 0.05); color: #cbd5e1; font-size: 0.95rem; line-height: 1.8; margin: 1rem 0; font-family: 'Inter', sans-serif;">
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Statut:</b> <span style="color: {'#4ade80' if status=='valid' else '#facc15' if status=='suspect' else '#f87171'};">{status_text}</span><br/>
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Dosage:</b> {val_dosage}<br/>
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Posologie:</b> {val_poso}<br/>
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Durée:</b> {val_duree}<br/>
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Prix Public:</b> {val_prix}<br/>
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Remboursable CNOPS:</b> {val_cnops}<br/>
                        <span style="color:#d4af37; margin-right:8px;">▪</span> <b>Type:</b> {val_type}
                    </div>
                    """

                    color = '#4ade80' if conf_class == 'high' else '#fbbf24' if conf_class == 'medium' else '#f87171'
                    conf_bar = (
                        f'<div class="conf-bar-bg">'
                        f'<div class="conf-bar-fill {conf_class}" style="width:{conf_pct}%"></div>'
                        f'</div>'
                        f'<div class="conf-label">'
                        f'<span>Confiance</span>'
                        f'<span style="color:{color};font-weight:700">{conf_pct:.0f}%</span>'
                        f'</div>'
                    )

                    card_html = (
                        f'<div class="med-card {conf_class}" style="animation-delay:{i * 0.1}s">'
                        f'<div class="med-card-header">'
                        f'<div><h3 class="med-name">💊 {display_name}</h3></div>'
                        f'{badge}'
                        f'</div>'
                        f'{list_html}'
                        f'{conf_bar}'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Section de correction manuelle interactive
                    with st.expander(f"⚙️ Action : Corriger ou Supprimer"):
                        c_edit1, c_edit2 = st.columns(2)
                        with c_edit1:
                            new_name = st.text_input("Nom du médicament", value=med.nom, key=f"edit_name_{i}")
                        with c_edit2:
                            new_dosage = st.text_input("Dosage", value=med.dosage or "", key=f"edit_dosage_{i}")
                        
                        col_btn1, col_btn2 = st.columns([1, 1])
                        with col_btn1:
                            if st.button("🔄 Vérifier", key=f"update_btn_{i}", use_container_width=True):
                                st.session_state["ocr_extraction"].medicaments[i].nom = new_name
                                st.session_state["ocr_extraction"].medicaments[i].dosage = new_dosage
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:
                                    st.experimental_rerun()
                        with col_btn2:
                            if st.button("🗑️ Supprimer", key=f"del_btn_{i}", use_container_width=True):
                                st.session_state["ocr_extraction"].medicaments.pop(i)
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:
                                    st.experimental_rerun()

            else:
                st.warning(
                    "Aucun médicament n'a pu être identifié. "
                    "Vous pouvez le(s) saisir manuellement ci-dessous."
                )

            # ── Section d'ajout manuel ──
            st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1); margin: 2rem 0;'/>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#f8fafc;'>➕ Ajouter un médicament manuellement</h4>", unsafe_allow_html=True)
            with st.form("add_med_form", clear_on_submit=True):
                col_ad1, col_ad2 = st.columns(2)
                with col_ad1:
                    add_name = st.text_input("Nom du médicament (ex: Doliprane)")
                with col_ad2:
                    add_dosage = st.text_input("Dosage (optionnel, ex: 1000mg)")
                add_submit = st.form_submit_button("Ajouter à la liste")
                if add_submit and add_name:
                    from engine.vision_ocr.vlm_extraction import Medicament
                    new_med = Medicament(nom=add_name, dosage=add_dosage, posologie=None, duree=None)
                    st.session_state["ocr_extraction"].medicaments.append(new_med)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()

            # ── Disclaimer final ──
            st.markdown("""
            <div class="ord-disclaimer">
                <p>
                    <b>⚠️ AVERTISSEMENT LÉGAL — MENTIONS OBLIGATOIRES :</b><br/><br/>
                    ① Les résultats de ce scanner sont générés par un système automatisé de reconnaissance
                    optique de caractères. Ils sont fournis <b>à titre informatif uniquement</b>.<br/>
                    ② Ce système <b>ne constitue pas un avis médical ni pharmaceutique</b>.<br/>
                    ③ <b>Consultez systématiquement votre médecin ou votre pharmacien</b>
                    pour la validation de toute ordonnance.<br/>
                    <span style="color:#64748b; font-size:0.78rem;">
                        Conformément à la réglementation marocaine, seul un professionnel de santé habilité peut prescrire
                        et délivrer des médicaments.
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
