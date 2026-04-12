"""
SHIFA AI — 📋 مسح الوصفة الطبية
═══════════════════════════════════
Analyse intelligente d'ordonnances via Vision AI (Gemini 3 Flash OpenRouter)
avec validation croisée CNOPS et medicament.ma.
"""

import streamlit as st
from PIL import Image
import io
import logging

from engine.vision_ocr.vlm_extraction import extract_from_image
import engine.vision_ocr.decision_engine as dec_engine

logger = logging.getLogger(__name__)

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="SHIFA AI — مسح الوصفة",
    page_icon="📋",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

    .ocr-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(0, 201, 167, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .ocr-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 30% 50%, rgba(0,201,167,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .ocr-header h2 {
        color: #00C9A7;
        font-family: 'Tajawal', sans-serif;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .ocr-header p {
        color: #94A3B8;
        font-family: 'Tajawal', sans-serif;
        font-size: 1rem;
    }

    .med-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(0, 201, 167, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .med-card:hover {
        border-color: rgba(0, 201, 167, 0.5);
        box-shadow: 0 4px 20px rgba(0, 201, 167, 0.1);
    }
    .med-card h4 {
        color: #00C9A7;
        font-family: 'Tajawal', sans-serif;
        margin-bottom: 0.5rem;
    }

    .status-valid {
        color: #22C55E;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
    }
    .status-suspect {
        color: #EAB308;
        background: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
    }
    .status-invalid {
        color: #EF4444;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def render_ocr_page():
    """Page principale de scan d'ordonnance via VLM."""

    # ── Header ───────────────────────────────────────────
    st.markdown("""
    <div class="ocr-header" dir="rtl">
        <h2>📋 مسح الوصفة الطبية (Vision AI)</h2>
        <p>استخراج ذكي مع تحقق من قاعدة CNOPS و medicament.ma</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────
    uploaded = st.file_uploader(
        "📸 ارفع صورة الوصفة",
        type=["jpg", "jpeg", "png", "webp"],
        help="تنسيقات مدعومة: JPG, JPEG, PNG, WEBP"
    )

    if not uploaded:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#64748B;" dir="rtl">
            <p style="font-size:3rem;">📋</p>
            <p style="font-family:'Tajawal',sans-serif; font-size:1.1rem;">
                ارفع صورة الوصفة الطبية لبدء التحليل
            </p>
            <p style="font-size:0.85rem; color:#475569;">
                مدعوم بواسطة Gemini 3 Flash للتعرف الدقيق
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load image ───────────────────────────────────────
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="📄 الوصفة الأصلية", width='stretch' if hasattr(st, 'container_width') else None, use_column_width=True)

    with col2:
        with st.spinner("⏳ جاري التحليل بواسطة الذكاء الاصطناعي..."):
            try:
                uploaded.seek(0)
                image_bytes = uploaded.getvalue()

                # Extraction via VLM
                extraction = extract_from_image(image_bytes)
                
                if not getattr(extraction, 'medicaments', []):
                    st.warning("⚠️ لم يتم اكتشاف أي أدوية في هذه الوصفة.")
                    final_results = {}
                else:
                    st.success(f"✅ تم استخراج {len(extraction.medicaments)} دواء (الثقة: {extraction.confiance_globale:.0%})")
                    # Validation
                    final_results = {}
                    for med in extraction.medicaments:
                        analysis = dec_engine.analyze_medication(
                            raw_name=med.nom,
                            raw_dosage=med.dosage,
                            vlm_confidence=extraction.confiance_globale,
                            posologie=med.posologie,
                            duree=med.duree,
                        )
                        key_name = analysis.get("corrected_name", med.nom)
                        final_results[key_name] = analysis

            except Exception as e:
                st.error(f"❌ خطأ في التحليل: {e}")
                logger.exception("VLM OCR pipeline error")
                return

        # ── Affichage Informations Générales ──────────────────
        medecin = getattr(extraction, "medecin", None)
        patient = getattr(extraction, "patient", None)
        
        if medecin or patient:
            st.markdown("### 📋 معلومات الوصفة", unsafe_allow_html=True)
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                if medecin:
                    nom_med = medecin.nom or "غير محدد"
                    spec = medecin.specialite or ""
                    st.info(f"👨‍⚕️ **طبيب:** {nom_med} {f'({spec})' if spec else ''}")
            with info_col2:
                if patient:
                    nom_pat = patient.nom or "غير محدد"
                    st.info(f"🧑 **مريض:** {nom_pat}")

        # ── Affichage des Médicaments ────────────────────────
        if final_results:
            st.markdown("### 💊 الأدوية المستخرجة")
            
            for med_name, info in final_results.items():
                status = info.get("status", "unknown")
                if status == "valid":
                    status_icon = "✅"
                    status_class = "status-valid"
                    status_text = "مؤكد"
                elif status == "suspect":
                    status_icon = "⚠️"
                    status_class = "status-suspect"
                    status_text = "مشتبه به"
                else:
                    status_icon = "❌"
                    status_class = "status-invalid"
                    status_text = "غير معروف"

                confidence = info.get("confidence", 0)
                
                with st.expander(f"{status_icon} **{med_name}** — (الثقة: {confidence*100:.0f}%)", expanded=(status == 'valid')):
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**الحالة:** <span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True)
                        st.markdown(f"**الجرعة (Dosage):** {info.get('dosage') or 'غير محدد'}")
                        st.markdown(f"**طريقة الأخذ (Posologie):** {info.get('posologie') or 'غير محدد'}")
                        st.markdown(f"**المدة (Durée):** {info.get('duree') or 'غير محدد'}")
                    with c_b:
                        price = info.get("price")
                        st.markdown(f"**الثمن (Prix Public):** {f'{price} DH' if price else 'غير متوفر'}")
                        remb = "نعم 💰" if info.get("remboursable") else "لا 🚫"
                        st.markdown(f"**التعويض (CNOPS Remboursable):** {remb}")
                        st.markdown(f"**النوع (Type):** {info.get('type') or 'N/A'}")

render_ocr_page()
