"""
Streamlit Kontrol Paneli Yöneticisi ve Sayfa Düzeni (Dashboard Page Manager).
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import streamlit as st

from .ai_modulleri import DashboardAIEngine
from .bilesenler import (
    metrik_kartlari_ciz,
    tespit_kutularini_ciz,
    sohbet_mesajlarini_ciz,
    guven_cubugu_ciz
)


def oturumu_baslat() -> DashboardAIEngine:
    """Session state içerisinde AI motorunu ve sohbet geçmişini başlatır."""
    if "ai_engine" not in st.session_state:
        st.session_state["ai_engine"] = DashboardAIEngine(device="cpu")
    if "sohbet_gecmisi" not in st.session_state:
        st.session_state["sohbet_gecmisi"] = [
            {"rol": "assistant", "icerik": "Merhaba! Ben kurumsal AI asistanınızım. Dokümanlar hakkında bana soru sorabilirsiniz.", "kaynaklar": []}
        ]
    return st.session_state["ai_engine"]


def dashboard_calistir():
    """Streamlit web uygulamasını render eder."""
    st.set_page_config(
        page_title="AI Kontrol Paneli - Day 36",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    ai_engine = oturumu_baslat()

    # -------------------------------------------------------------
    # 1. Yan Menü (Sidebar) Yapılandırması
    # -------------------------------------------------------------
    with st.sidebar:
        st.image("https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit", use_container_width=True)
        st.title("⚙️ Model Ayarları")
        aktif_model = st.selectbox("Hedef Model", ["MiniAIEngine (PyTorch)", "VisionDetector v1", "MiniRAG v2"])
        guven_esigi = st.slider("Görsel Güven Eşiği (Threshold)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
        top_k_rag = st.slider("RAG Parça Sayısı (Top-K)", min_value=1, max_value=5, value=2)

        st.divider()
        if st.button("🔄 Oturum Geçmişini Temizle"):
            st.session_state["sohbet_gecmisi"] = []
            st.rerun()

    # -------------------------------------------------------------
    # 2. Üst Bilgi ve Metrikler
    # -------------------------------------------------------------
    st.title("🧠 İnteraktif Çoklu Görev AI Kontrol Paneli")
    st.caption("Doğal Dil İşleme, Bilgisayarlı Görü ve RAG Doküman Asistanı Tek Bir Yönetim Panelinde.")

    gecikmeler = ai_engine.gecikme_gecmisi
    ort_gecikme = sum(gecikmeler) / len(gecikmeler) if gecikmeler else 0.0
    metrik_kartlari_ciz(ai_engine.istek_sayisi, ort_gecikme, aktif_model)

    st.divider()

    # -------------------------------------------------------------
    # 3. Ana Sekmeler (Tabs)
    # -------------------------------------------------------------
    sekme1, sekme2, sekme3, sekme4 = st.tabs([
        "📝 Metin Analizi & Sınıflandırma",
        "🖼️ Bilgisayarlı Görü & Kusur Tespiti",
        "💬 RAG Doküman Asistanı",
        "📈 Sistem Telemetrisi"
    ])

    # SEKME 1: Metin Analizi
    with sekme1:
        st.subheader("Doğal Dil İşleme & Konu Sınıflandırma")
        col_t1, col_t2 = st.columns([1.5, 1])

        with col_t1:
            varsayilan_metin = "Evrişimli sinir ağları ve YOLO mimarisi ile görsel nesne tespiti"
            kullanici_metni = st.text_area("Analiz Edilecek Metin:", value=varsayilan_metin, height=120)
            if st.button("🚀 Metni Sınıflandır"):
                sonuc = ai_engine.metin_analiz_et(kullanici_metni)
                st.session_state["son_metin_sonucu"] = sonuc

        with col_t2:
            if "son_metin_sonucu" in st.session_state:
                res = st.session_state["son_metin_sonucu"]
                st.success(f"🎯 **Tahmin:** {res['tahmin_sinifi']}")
                st.info(f"⚡ **Çıkarım Gecikmesi:** {res['gecikme_ms']:.2f} ms")
                guven_cubugu_ciz(res["olasiliklar"])
                with st.expander("🔢 64-Boyutlu Embedding Vektörü"):
                    st.json(res["embedding"][:8])

    # SEKME 2: Bilgisayarlı Görü
    with sekme2:
        st.subheader("Görsel Analizi & Kumaş Kusur Tespiti")
        yuklenen_dosya = st.file_uploader("Bir görsel yükleyin (PNG / JPG):", type=["png", "jpg", "jpeg"])

        if yuklenen_dosya is not None:
            gorsel = Image.open(yuklenen_dosya)
        else:
            # Örnek sentetik görsel oluştur
            sentetik_dizi = np.ones((300, 400, 3), dtype=np.uint8) * 200
            gorsel = Image.fromarray(sentetik_dizi)

        col_v1, col_v2 = st.columns([1.2, 1])
        with col_v1:
            gorsel_sonuc = ai_engine.gorsel_analiz_et(gorsel, guven_esigi=guven_esigi)
            cizili_gorsel = tespit_kutularini_ciz(gorsel, gorsel_sonuc["tespitler"])
            st.image(cizili_gorsel, caption=f"İşlenmiş Görsel ({gorsel.size[0]}x{gorsel.size[1]})", use_container_width=True)

        with col_v2:
            st.write(f"**🔍 Tespit Edilen Kusur Sayısı:** {len(gorsel_sonuc['tespitler'])}")
            st.dataframe(gorsel_sonuc["tespitler"], use_container_width=True)
            renk = gorsel_sonuc["baskin_renk"]
            st.write(f"**🎨 Baskın RGB:** `{renk}`")

    # SEKME 3: RAG Sohbet Asistanı
    with sekme3:
        st.subheader("Doküman Tabanlı Soru-Cevap Motoru")
        sohbet_mesajlarini_ciz(st.session_state["sohbet_gecmisi"])

        kullanici_sorusu = st.chat_input("Bir soru yazın (ör: YOLO nedir?)...")
        if kullanici_sorusu:
            st.session_state["sohbet_gecmisi"].append({"rol": "user", "icerik": kullanici_sorusu})
            rag_yanit = ai_engine.rag_soru_sor(kullanici_sorusu, top_k=top_k_rag)
            st.session_state["sohbet_gecmisi"].append({
                "rol": "assistant",
                "icerik": rag_yanit["yanit"],
                "kaynaklar": rag_yanit["kaynaklar"]
            })
            st.rerun()

    # SEKME 4: Telemetri
    with sekme4:
        st.subheader("Sistem & Çıkarım Telemetrisi")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**Gecikme Geçmişi (ms):**")
            if ai_engine.gecikme_gecmisi:
                st.line_chart(ai_engine.gecikme_gecmisi)
            else:
                st.info("Henüz çıkarım isteği yapılmadı.")
        with col_m2:
            st.write("**Donanım Durumu:**")
            st.json({
                "Cihaz": str(ai_engine.device),
                "Model_Mimari": "MiniAIEngine (Attention + FC)",
                "Toplam_Istek": ai_engine.istek_sayisi
            })
