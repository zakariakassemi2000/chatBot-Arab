"""
SHIFA AI — 📋 مسح الوصفة الطبية
═══════════════════════════════════
Extraction OCR des ordonnances médicales
Engine: docTR (CPU) | Donut Medical (GPU) | Tesseract (fallback)
"""

import streamlit as st
from PIL import Image
import requests.utils
import logging

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
    .med-card .detail {
        color: #CBD5E1;
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    .med-card .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-found {
        background: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-notfound {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-nocheck {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        flex: 1;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(0, 201, 167, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box .number {
        font-size: 2rem;
        font-weight: 700;
        color: #00C9A7;
    }
    .stat-box .label {
        color: #94A3B8;
        font-size: 0.85rem;
        font-family: 'Tajawal', sans-serif;
    }

    .raw-text-box {
        background: #0f172a;
        border: 1px solid rgba(0, 201, 167, 0.15);
        border-radius: 8px;
        padding: 1rem;
        color: #E2E8F0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
        direction: ltr;
        text-align: left;
    }

    .warning-box {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #EAB308;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .info-box {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #3B82F6;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def render_ocr_page():
    """Page principale de scan d'ordonnance."""

    # ── Header ───────────────────────────────────────────
    st.markdown("""
    <div class="ocr-header" dir="rtl">
        <h2>📋 مسح الوصفة الطبية</h2>
        <p>استخراج المعلومات من الوصفة — لا تفسير طبي</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Options ──────────────────────────────────────────
    col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 1])
    with col_opt1:
        engine = st.selectbox(
            "🔧 محرك OCR",
            [
                "docTR (موصى به — CPU)",
                "Donut Medical (GPU مفضل)",
            ],
            index=0,
            help="docTR هو الأسرع والأدق على الوصفات المطبوعة بالفرنسية"
        )
    with col_opt2:
        verify = st.checkbox(
            "🌐 التحقق على medicament.ma",
            value=True,
            help="يتحقق من وجود الأدوية في قاعدة بيانات medicament.ma"
        )
    with col_opt3:
        st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    # ── Upload ───────────────────────────────────────────
    uploaded = st.file_uploader(
        "📷 ارفع صورة الوصفة",
        type=["jpg", "jpeg", "png"],
        help="تنسيقات مدعومة: JPG, JPEG, PNG"
    )

    if not uploaded:
        # Placeholder when no image uploaded
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#64748B;" dir="rtl">
            <p style="font-size:3rem;">📋</p>
            <p style="font-family:'Tajawal',sans-serif; font-size:1.1rem;">
                ارفع صورة الوصفة الطبية لبدء التحليل
            </p>
            <p style="font-size:0.85rem; color:#475569;">
                يدعم الوصفات المطبوعة بالفرنسية والعربية
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load image ───────────────────────────────────────
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="📄 الوصفة الأصلية", use_container_width=True)

    with col2:
        with st.spinner("⏳ جاري التحليل بواسطة الذكاء الاصطناعي..."):
            try:
                from engine.ocr_ordonnance import get_ocr

                use_donut = "Donut" in engine
                ocr = get_ocr(
                    use_donut=use_donut,
                    verify_online=verify
                )
                res = ocr.analyser(image)

            except ImportError as e:
                st.error(f"❌ خطأ في تحميل محرك OCR: {e}")
                st.info("💡 تأكد من تثبيت المتطلبات: `pip install python-doctr[torch] beautifulsoup4 rapidfuzz`")
                return
            except Exception as e:
                st.error(f"❌ خطأ في التحليل: {e}")
                logger.exception("OCR pipeline error")
                return

        # ── Success banner ───────────────────────────────
        st.success(f"✅ تم التحليل بنجاح — محرك: **{res.engine_utilise}**")

        # ── Stats row ────────────────────────────────────
        lang_display = {
            "fr": "🇫🇷 فرنسية",
            "ar": "🇲🇦 عربية",
            "mixte": "🌍 مختلطة",
            "inconnu": "❓ غير محدد"
        }

        n_meds = len(res.medicaments)
        lang_str = lang_display.get(res.langue_detectee, res.langue_detectee)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box">
                <div class="number">{n_meds}</div>
                <div class="label">💊 أدوية مكتشفة</div>
            </div>
            <div class="stat-box">
                <div class="number">{lang_str}</div>
                <div class="label">🌐 اللغة</div>
            </div>
            <div class="stat-box">
                <div class="number">{res.engine_utilise.split('(')[0].strip()}</div>
                <div class="label">🔧 المحرك</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Medications ──────────────────────────────────
        if res.medicaments:
            st.markdown(f"### 💊 الأدوية المستخرجة ({n_meds})")

            for i, med in enumerate(res.medicaments, 1):
                # Badge de vérification
                if med.verification_ma:
                    vm = med.verification_ma
                    if vm.get("found"):
                        badge_html = '<span class="badge badge-found">🟢 موجود في medicament.ma</span>'
                    else:
                        badge_html = '<span class="badge badge-notfound">🔴 غير موجود</span>'
                else:
                    badge_html = '<span class="badge badge-nocheck">⚪ لم يتم التحقق</span>'

                title = med.dci or med.nom_brut
                with st.expander(f"**{i}. {title}**", expanded=(i <= 3)):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="med-card" dir="rtl">
                            <h4>{title}</h4>
                            {badge_html}
                        """, unsafe_allow_html=True)

                        if med.dci:
                            st.markdown(f"**DCI :** {med.dci}")
                        if med.nom_commercial:
                            st.markdown(f"**الاسم التجاري :** {med.nom_commercial}")
                        if med.dosage:
                            st.markdown(f"**الجرعة :** {med.dosage}")
                        if med.posologie:
                            st.markdown(f"**طريقة الأخذ :** {med.posologie}")
                        if med.duree:
                            st.markdown(f"**المدة :** {med.duree}")

                        st.markdown(f"**الثقة OCR :** {med.confidence_ocr:.0%}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with c2:
                        if med.verification_ma:
                            vm = med.verification_ma
                            if vm.get("found"):
                                st.success("✅ موجود على medicament.ma")
                                for m in vm.get("medicaments", [])[:3]:
                                    url = m.get("url", "")
                                    nom = m.get("nom", "")
                                    if url and nom:
                                        st.markdown(f"🔗 [{nom}]({url})")
                            else:
                                st.warning("⚠️ غير موجود في القاعدة")
                                first_word = med.nom_brut.split()[0] if med.nom_brut else ""
                                search_url = f"https://medicament.ma/?s={requests.utils.quote(first_word)}"
                                st.markdown(f"[🔍 بحث يدوي على medicament.ma]({search_url})")
                        else:
                            st.info("ℹ️ التحقق عبر الإنترنت معطل")
        else:
            st.warning("⚠️ لم يتم اكتشاف أي أدوية في هذه الوصفة")

        # ── Raw text ─────────────────────────────────────
        with st.expander("📄 النص الكامل المستخرج"):
            if res.texte_brut:
                st.markdown(
                    f'<div class="raw-text-box">{res.texte_brut}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("لا يوجد نص مستخرج")

        # ── Warnings ─────────────────────────────────────
        for w in res.avertissements:
            if "⚠️" in w:
                st.markdown(f'<div class="warning-box">{w}</div>',
                            unsafe_allow_html=True)
            elif "ℹ️" in w:
                st.markdown(f'<div class="info-box">{w}</div>',
                            unsafe_allow_html=True)
            else:
                st.info(w)


# ── Run ──────────────────────────────────────────────────
render_ocr_page()
