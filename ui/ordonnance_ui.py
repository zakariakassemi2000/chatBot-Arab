# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════
  SHIFA AI — ماسح الوصفات الطبية المحسن (Prescription Scanner)
  واجهة مستخدم احترافية بالكامل باللغة العربية مع دعم RTL والتبويب الفاتح
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import logging
import io

logger = logging.getLogger("shifa.ordonnance_ui")


def _inject_ordonnance_css():
    """Injects premium CSS styles for the prescription module."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

        .material-symbols-rounded {
            font-family: 'Material Symbols Rounded' !important;
            font-weight: normal;
            font-style: normal;
            font-size: 20px;
            display: inline-block;
            line-height: 1;
            text-transform: none;
            letter-spacing: normal;
            word-wrap: normal;
            white-space: nowrap;
            direction: ltr;
            -webkit-font-smoothing: antialiased;
        }

        /* ═══ Entrada Animation ═══ */
        @keyframes ordFadeIn {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .ord-animate {
            animation: ordFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        /* ═══ Medical Disclaimer Alert ═══ */
        .ord-disclaimer {
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-right: 4px solid #f59e0b;
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            animation: ordFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .ord-disclaimer p {
            color: #b45309;
            font-size: 0.95rem;
            line-height: 1.8;
            margin: 0;
        }
        .ord-disclaimer b {
            color: #d97706;
        }

        /* ═══ File Uploader Light Container ═══ */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(8, 145, 178, 0.25) !important;
            background: var(--shifa-primary-light) !important;
            border-radius: 16px !important;
            padding: 2.5rem 1.5rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: var(--shifa-primary) !important;
            background: rgba(8, 145, 178, 0.12) !important;
        }

        /* ═══ Medication Card styling ═══ */
        .med-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: ordFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .med-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 25px rgba(8, 145, 178, 0.15);
            border-color: rgba(8, 145, 178, 0.3);
        }

        /* Right status accent line (RTL) */
        .med-card::before {
            content: "";
            position: absolute;
            top: 0; right: 0;
            width: 5px; height: 100%;
            border-radius: 0 16px 16px 0;
        }
        .med-card.high::before   { background: linear-gradient(to bottom, #10b981, #34d399); }
        .med-card.medium::before { background: linear-gradient(to bottom, #f59e0b, #fbbf24); }
        .med-card.low::before    { background: linear-gradient(to bottom, #ef4444, #f87171); }

        /* Card header */
        .med-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            direction: rtl;
        }
        .med-name {
            font-size: 1.35rem;
            font-weight: 800;
            color: #1e293b;
            margin: 0;
            font-family: 'Cairo', sans-serif;
        }

        /* Badges */
        .med-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
        }
        .med-badge.found {
            background: rgba(16, 185, 129, 0.15);
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .med-badge.suspect {
            background: rgba(245, 158, 11, 0.15);
            color: #b45309;
            border: 1px solid #fcd34d;
        }
        .med-badge.notfound {
            background: rgba(239, 68, 68, 0.15);
            color: #991b1b;
            border: 1px solid #fca5a5;
        }

        /* Confidence bar */
        .conf-bar-bg {
            background: #f1f5f9;
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
            margin-top: 1rem;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .conf-bar-fill.high   { background: linear-gradient(90deg, #10b981, #34d399); }
        .conf-bar-fill.medium { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
        .conf-bar-fill.low    { background: linear-gradient(90deg, #ef4444, #f87171); }

        .conf-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 6px;
            font-family: 'Cairo', sans-serif;
        }

        /* Global score widget */
        .score-global {
            background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
            border: 1px solid rgba(8, 145, 178, 0.15);
            border-radius: 20px;
            padding: 1.8rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(8, 145, 178, 0.05);
        }
        .score-number {
            font-size: 3.8rem;
            font-weight: 900;
            margin: 0.5rem 0;
            font-family: 'Cairo', sans-serif;
        }
        .score-number.high   { color: #059669; }
        .score-number.medium { color: #d97706; }
        .score-number.low    { color: #dc2626; }
    </style>
    <script>
    const _shifaInitLocalOrdo = () => {
        const doc = window.parent.document || document;
        doc.querySelectorAll('button').forEach(btn => {
            if (btn.textContent.trim() === 'Browse files') btn.textContent = 'تصفح الملفات';
        });
        doc.querySelectorAll('p, span, div, small, label').forEach(el => {
            if (el.childElementCount === 0) {
                let t = el.textContent.trim();
                if (t === 'Drag and drop file here') el.textContent = 'اسحب وأفلت الملف هنا';
                if (t.startsWith('Limit') || (t.includes('Limit') && t.includes('per file'))) {
                    el.textContent = 'الحد الأقصى 10 ميغا • JPG, JPEG, PNG, WEBP';
                }
            }
        });
    };
    const _obsLocalOrdo = new MutationObserver(_shifaInitLocalOrdo);
    _obsLocalOrdo.observe(window.parent.document.body || document.body, {childList: true, subtree: true});
    setTimeout(_shifaInitLocalOrdo, 500);
    setTimeout(_shifaInitLocalOrdo, 2000);
    </script>
    """, unsafe_allow_html=True)


def _get_confidence_class(score: float) -> str:
    """Returns the CSS confidence class based on score."""
    if score >= 75:
        return "high"
    elif score >= 50:
        return "medium"
    return "low"


def render_ordonnance_page():
    """
    Renders the beautiful, fully-Arabic, responsive prescription scanner page.
    """
    _inject_ordonnance_css()

    # ── Page Header ──
    st.markdown(
        '<div class="moroccan-title" style="font-size:2.8rem; font-family:\'Cairo\';">📋 ماسح الوصفات الطبية</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="moroccan-subtitle" style="font-family:\'Cairo\';">تحليل وقراءة الوصفات الطبية بالذكاء الاصطناعي ومطابقتها مع الدليل الدوائي المغربي</div>',
        unsafe_allow_html=True
    )


    # ── Prescription Image Source ──
    with st.container(border=True):
        st.markdown(
            "<h4 style='color:#1e293b; font-family:\"Cairo\"; display:flex; align-items:center; gap:8px;'><span class='material-symbols-rounded' style='color:#0891B2;'>photo_camera</span> تحميل أو التقاط صورة الوصفة</h4>",
            unsafe_allow_html=True
        )

        tab_upload, tab_camera = st.tabs(["📁 تحميل ملف الوصفة", "📸 التقاط صورة مباشرة"])

        image = None

        with tab_upload:
            uploaded_file = st.file_uploader(
                "اسحب وأفلت صورة الوصفة الطبية هنا",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                help="الصيغ المدعومة: PNG, JPG, JPEG, BMP, TIFF, WEBP",
                key="ord_upload"
            )
            if uploaded_file:
                from PIL import Image as PILImage
                image = PILImage.open(uploaded_file)

        with tab_camera:
            camera_photo = st.camera_input(
                "التقط صورة واضحة للوصفة الطبية",
                key="ord_camera"
            )
            if camera_photo:
                from PIL import Image as PILImage
                image = PILImage.open(camera_photo)

    # ── Image Upload Preview & Analysis Actions ──
    if image is not None:
        with st.container(border=True):
            col_img, col_action = st.columns([1.2, 1])

            with col_img:
                st.image(image, caption="صورة الوصفة الطبية المحملة", width="stretch")

            with col_action:
                st.markdown("""
                <div style="padding: 1rem 0; text-align: right; direction: rtl;">
                    <h4 style="color: #1e293b; margin-bottom: 0.8rem; font-family: 'Cairo'; display:flex; align-items:center; gap:6px;"><span class="material-symbols-rounded" style="color:#0891B2;">psychology</span> معالجة وقراءة الذكاء الاصطناعي</h4>
                    <p style="color: #475569; font-size: 0.9rem; line-height: 1.8; font-family: 'Cairo';">
                        سيقوم النظام بتنفيذ المهام التالية تلقائياً:<br/>
                        ✓ تحسين جودة وتفتيح الصورة لتحسين التعرف<br/>
                        ✓ قراءة واستخراج النصوص الطبية المكتوبة<br/>
                        ✓ استخلاص أسماء الأدوية والجرعات والمدد الموصوفة<br/>
                        ✓ مطابقة البيانات المستخرجة مع الدليل الدوائي الوطني المغربي<br/>
                        ✓ جلب تفاصيل الأسعار ونسب تعويض التأمين الصحي (CNOPS)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                analyze_btn = st.button(
                    "🚀 بدء التحليل والتشخيص الذكي",
                    type="primary",
                    width="stretch",
                    key="ord_analyze"
                )

        if analyze_btn:
            with st.status("🔬 يجري الآن قراءة وتحليل الوصفة الطبية...", expanded=True) as status_box:
                try:
                    from engine.vision_ocr.vlm_extraction import extract_from_image
                    
                    st.write("⏳ جاري تهيئة الصورة ومعالجتها رقمياً...")
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format or 'JPEG')
                    image_bytes = img_byte_arr.getvalue()
                    
                    st.write("🧠 جاري تشغيل نموذج الرؤية الاصطناعية (VLM) لاستخراج أسماء الأدوية والجرعات...")
                    extraction = extract_from_image(image_bytes)
                    
                    st.write("📚 جاري مطابقة النتائج مع الدليل الدوائي والتحقق من الأسعار والتعويض...")
                    st.session_state["ocr_extraction"] = extraction
                    
                    status_box.update(label="✅ اكتمل فحص الوصفة الطبية بنجاح!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status_box.update(label="❌ فشل تحليل الوصفة الطبية", state="error", expanded=True)
                    st.error(f"حدث خطأ أثناء الفحص الذكي: {e}")
                    logger.error(f"OCR error: {e}", exc_info=True)
                    return

        # ── Display Results & Interactive Adjustments ──
        if st.session_state.get("ocr_extraction") is not None:
            extraction = st.session_state["ocr_extraction"]
            import engine.vision_ocr.decision_engine as dec_engine
            
            score_global = extraction.confiance_globale * 100
            score_class = _get_confidence_class(score_global)
            n_meds = len(extraction.medicaments) if hasattr(extraction, 'medicaments') and extraction.medicaments else 0

            st.markdown(f"""
            <div class="score-global ord-animate" style="direction: rtl;">
                <div style="color: #64748b; font-size: 1rem; font-weight: 700; font-family: 'Cairo';">
                    🎯 مؤشر دقة المطابقة الإجمالي
                </div>
                <div class="score-number {score_class}">
                    {score_global:.0f}%
                </div>
                <div style="color: #475569; font-size: 0.9rem; font-family: 'Cairo'; display: flex; align-items: center; justify-content: center; gap: 6px;">
                    <span class="material-symbols-rounded" style="font-size: 18px;">medication</span>
                    <span>تم تحديد {n_meds} دواء/أدوية بنجاح</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Doctor & Patient Info Panel ──
            medecin = getattr(extraction, "medecin", None)
            patient = getattr(extraction, "patient", None)
            
            if (medecin and (medecin.nom or medecin.specialite)) or (patient and patient.nom):
                st.markdown("<h3 style='color:#1e293b; margin: 1.5rem 0 1rem; font-family:\"Cairo\"; font-size:1.3rem; display: flex; align-items: center; gap: 8px;'><span class='material-symbols-rounded' style='color:#0891B2;'>contact_page</span> بيانات الطبيب والمريض المستخرجة</h3>", unsafe_allow_html=True)
                col_doc, col_pat = st.columns(2)
                with col_doc:
                    if medecin and (medecin.nom or medecin.specialite):
                        nom_med = medecin.nom or "غير محدد"
                        spec_med = medecin.specialite or "غير محدد"
                        st.markdown(f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; direction: rtl; text-align: right; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                            <div style="display: flex; align-items: center; gap: 8px; color: #0891B2; font-weight: 700; margin-bottom: 6px; font-family: 'Cairo';">
                                <span class="material-symbols-rounded">medical_services</span>
                                <b>بيانات الطبيب المعالج:</b>
                            </div>
                            <div style="color: #1e293b; font-size: 0.95rem; margin-bottom: 4px; font-family: 'Cairo';">الاسم: {nom_med}</div>
                            <div style="color: #475569; font-size: 0.85rem; font-family: 'Cairo';">التخصص: {spec_med}</div>
                        </div>
                        """, unsafe_allow_html=True)
                with col_pat:
                    if patient and patient.nom:
                        nom_pat = patient.nom or "غير محدد"
                        st.markdown(f"""
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem; direction: rtl; text-align: right; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                            <div style="display: flex; align-items: center; gap: 8px; color: #0891B2; font-weight: 700; margin-bottom: 6px; font-family: 'Cairo';">
                                <span class="material-symbols-rounded">person</span>
                                <b>بيانات المستفيد (المريض):</b>
                            </div>
                            <div style="color: #1e293b; font-size: 0.95rem; margin-bottom: 4px; font-family: 'Cairo';">الاسم: {nom_pat}</div>
                            <div style="color: #475569; font-size: 0.85rem; font-family: 'Cairo';">الملف الشخصي: مريض مسجل</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Medications Grid Display ──
            if n_meds > 0:
                st.markdown("<h3 style='color:#1e293b; margin: 2rem 0 1rem; font-family:\"Cairo\"; font-size:1.4rem; display: flex; align-items: center; gap: 8px;'><span class='material-symbols-rounded' style='color:#0891B2;'>list_alt</span> الأدوية المكتشفة في الوصفة</h3>", unsafe_allow_html=True)
                
                cols = st.columns(2)
                for i, med in enumerate(extraction.medicaments):
                    col_idx = i % 2
                    with cols[col_idx]:
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
                            badge = '<span class="med-badge found"><span class="material-symbols-rounded" style="font-size: 14px; margin-left: 4px;">check_circle</span>مطابق وموثق</span>'
                        elif status == "suspect":
                            badge = '<span class="med-badge suspect"><span class="material-symbols-rounded" style="font-size: 14px; margin-left: 4px;">warning</span>غير مؤكد</span>'
                        else:
                            badge = '<span class="med-badge notfound"><span class="material-symbols-rounded" style="font-size: 14px; margin-left: 4px;">cancel</span>غير مسجل</span>'

                        display_name = analysis.get("corrected_name", med.nom)
                        
                        val_dosage = analysis.get('dosage') or "غير محدد"
                        val_poso = analysis.get('posologie') or "غير محدد"
                        val_duree = analysis.get('duree') or "غير محدد"
                        val_prix = f"{analysis['price']} درهم" if analysis.get('price') else "غير متوفر"
                        val_cnops = "نعم (مسترجع CNOPS) ✅" if analysis.get("remboursable") else "لا (غير مسترجع CNOPS) 🚫"
                        
                        med_type = analysis.get("type") or "Unknown"
                        if med_type.lower() == "princeps":
                            val_type = "دواء أصيل (Princeps)"
                        elif med_type.lower() in ["generique", "générique"]:
                            val_type = "دواء جنيس (Générique)"
                        else:
                            val_type = "غير محدد"

                        list_html = f"""
                        <div class="med-info-list" style="background: #f8fafc; border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; color: #334155; font-size: 0.9rem; line-height: 1.8; margin: 1rem 0; direction: rtl; text-align: right; font-family: 'Cairo', sans-serif;">
                            <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                                <span><b>📌 حالة التطابق:</b></span>
                                <span style="color: {'#059669' if status=='valid' else '#d97706' if status=='suspect' else '#dc2626'}; font-weight: 700;">
                                    {'مطابق وموثق' if status=='valid' else 'غير مؤكد' if status=='suspect' else 'غير مسجل'}
                                </span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                                <span><b>💊 الجرعة المحددة:</b></span>
                                <span style="color:#1e293b;">{val_dosage}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                                <span><b>🔄 طريقة الاستعمال:</b></span>
                                <span style="color:#1e293b;">{val_poso}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                                <span><b>📅 مدة العلاج:</b></span>
                                <span style="color:#1e293b;">{val_duree}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                                <span><b>💰 السعر للعموم:</b></span>
                                <span style="color:#1e293b; font-weight:700;">{val_prix}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom: 6px;">
                                <span><b>🛡️ تغطية CNOPS:</b></span>
                                <span style="color:#1e293b;">{val_cnops}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <span><b>📦 نوع الدواء:</b></span>
                                <span style="color:#1e293b;">{val_type}</span>
                            </div>
                        </div>
                        """

                        color = '#10b981' if conf_class == 'high' else '#f59e0b' if conf_class == 'medium' else '#ef4444'
                        conf_bar = (
                            f'<div class="conf-bar-bg">'
                            f'<div class="conf-bar-fill {conf_class}" style="width:{conf_pct}%"></div>'
                            f'</div>'
                            f'<div class="conf-label" style="direction: rtl; text-align: right;">'
                            f'<span>نسبة مطابقة المادة الفعالة</span>'
                            f'<span style="color:{color};font-weight:700">{conf_pct:.0f}%</span>'
                            f'</div>'
                        )

                        card_html = (
                            f'<div class="med-card {conf_class}" style="animation-delay:{i * 0.1}s; direction: rtl; text-align: right;">'
                            f'<div class="med-card-header" style="direction: rtl;">'
                            f'<div><h3 class="med-name" style="font-family: \'Cairo\'; display:flex; align-items:center; gap:6px;"><span class="material-symbols-rounded" style="color:#0891B2;">healing</span> {display_name}</h3></div>'
                            f'{badge}'
                            f'</div>'
                            f'{list_html}'
                            f'{conf_bar}'
                            f'</div>'
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Interactive form for correction or deletion
                        with st.expander(f"⚙️ خيارات : تصحيح أو حذف الدواء"):
                            st.markdown("<p style='font-size:0.85rem; color:#475569; margin-bottom:8px; font-family: \"Cairo\";'>تعديل يدوي لبيانات الدواء المستخرج:</p>", unsafe_allow_html=True)
                            new_name = st.text_input("اسم الدواء", value=med.nom, key=f"edit_name_{i}")
                            new_dosage = st.text_input("الجرعة (مثال: 1000mg)", value=med.dosage or "", key=f"edit_dosage_{i}")
                            
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button("🔄 تحديث البيانات", key=f"update_btn_{i}", width="stretch"):
                                    st.session_state["ocr_extraction"].medicaments[i].nom = new_name
                                    st.session_state["ocr_extraction"].medicaments[i].dosage = new_dosage
                                    st.rerun()
                            with col_btn2:
                                if st.button("🗑️ حذف الدواء", key=f"del_btn_{i}", width="stretch"):
                                    st.session_state["ocr_extraction"].medicaments.pop(i)
                                    st.rerun()

            else:
                st.warning(
                    "لم يتم التعرف على أي أدوية بالوصفة. "
                    "يرجى مراجعة جودة الصورة وإضاءتها، أو إضافة الأدوية يدوياً بالأسفل."
                )

            # ── Manual Add Section ──
            st.markdown("<hr style='border:1px solid #e2e8f0; margin: 2.5rem 0;'/>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#1e293b; font-family:\"Cairo\"; display:flex; align-items:center; gap:8px;'><span class='material-symbols-rounded' style='color:#0891B2;'>add_circle</span> إضافة دواء يدوياً للقائمة</h4>", unsafe_allow_html=True)
            
            with st.form("add_med_form", clear_on_submit=True):
                col_ad1, col_ad2 = st.columns(2)
                with col_ad1:
                    add_name = st.text_input("اسم الدواء (مثال: Doliprane)", placeholder="أدخل اسم الدواء هنا...")
                with col_ad2:
                    add_dosage = st.text_input("الجرعة / التركيز (مثال: 1000mg)", placeholder="مثال: 1g, 500mg...")
                
                add_submit = st.form_submit_button("➕ إضافة الدواء للقائمة", width="stretch")
                if add_submit and add_name:
                    from engine.vision_ocr.vlm_extraction import Medicament
                    new_med = Medicament(nom=add_name, dosage=add_dosage, posologie=None, duree=None)
                    st.session_state["ocr_extraction"].medicaments.append(new_med)
                    st.rerun()

            # ── Final Legal Disclaimer ──
            st.markdown("""
            <div class="ord-disclaimer" style="border: 1px solid #fee2e2; border-right: 4px solid #ef4444; margin-top: 3rem; background: #fef2f2;">
                <p style="font-family: 'Cairo', sans-serif; line-height: 1.8; color: #991b1b;">
                    <b>⚠️ إخلاء مسؤولية قانوني — مقتضيات تنظيمية هامة:</b><br/><br/>
                    ١. النتائج المستخرجة عبر نظام قراءة الصور والتعرف الضوئي بالذكاء الاصطناعي هي نتائج آلية استرشادية، وتُقدم <b>بصفة إعلامية وتثقيفية فقط</b>.<br/>
                    ٢. لا يُشكل هذا النظام ولا تقاريره أي <b>بديل عن الاستشارة الطبية أو الصيدلانية الرسمية</b> ولا يُعتمد عليه بمفرده لتناول أي أدوية.<br/>
                    ٣. <b>يجب عليك مراجعة الطبيب المعالج أو الصيدلاني المؤهل بشكل مباشر</b> لتأكيد ملاءمة الوصفة وصرف الدواء بشكل صحيح وآمن.<br/>
                    <span style="color:#475569; font-size:0.8rem;">
                        تماشياً مع المقتضيات القانونية المعمول بها بالمملكة المغربية، فإن كتابة ووصف الأدوية وتوجيه الجرعات هي من الصلاحيات والمسؤوليات الحصرية للأطباء والصيادلة المؤهلين الحاملين للتراخيص المهنية المطلوبة.
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
