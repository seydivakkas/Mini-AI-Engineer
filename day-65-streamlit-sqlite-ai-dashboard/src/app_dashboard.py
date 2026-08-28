"""
Streamlit SQLite AI Model Çıkarım ve Yönetim Paneli (Streamlit Application).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Proje kök dizinini ekle
MEVCUT_DIZIN = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.veritabani_yoneticisi import AIVeritabaniYoneticisi
from src.analiz_motoru import AITelemetriAnalizci

st.set_page_config(
    page_title="AI Çıkarım ve Model Yönetim Paneli",
    page_icon="🤖",
    layout="wide"
)

DB_PATH = os.path.join(MEVCUT_DIZIN, "ai_yonetim.db")
db = AIVeritabaniYoneticisi(db_yolu=DB_PATH)

st.title("🤖 AI Model Çıkarım Logları ve Yönetim Paneli")
st.markdown("**SQLite Destekli Kalıcı CRUD, Telemetri İzleme ve İnsan Denetimi (Human-in-the-Loop)**")

# -------------------------------------------------------------
# Kenar Çubuğu (Sidebar) Filtreleri
# -------------------------------------------------------------
st.sidebar.header("🔍 Filtreleme ve Kontrol")
model_secimi = st.sidebar.selectbox(
    "Model Seçiniz:",
    ["Tümü", "YOLOv8x-Vision", "MiniViT-Embedder", "Defect-Detector-V2"]
)
min_guven = st.sidebar.slider("Minimum Ortalama Güven Skoru:", 0.0, 1.0, 0.0, 0.05)
kayit_limiti = st.sidebar.slider("Gösterilecek Kayıt Limiti:", 10, 500, 100, 10)

if st.sidebar.button("🎲 Sentetik Veri Ekle (50 Kayıt)"):
    AITelemetriAnalizci.sentetik_veri_doldur(db, kayit_sayisi=50)
    st.sidebar.success("50 yeni sentetik çıkarım logu eklendi!")

# -------------------------------------------------------------
# 1. Üst KPI Metrik Kartları
# -------------------------------------------------------------
stats = db.genel_istatistikleri_al()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Toplam İstek", f"{stats['toplam_istek']:,}")
with col2:
    st.metric("Ortalama Gecikme", f"{stats['ortalama_gecikme_ms']} ms")
with col3:
    st.metric("Ortalama Güven", f"%{stats['ortalama_guven']*100:.1f}")
with col4:
    st.metric("Toplam Tespit", f"{stats['toplam_tespit']:,}")

st.divider()

# -------------------------------------------------------------
# 2. Sekmeler (Tabs)
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Çıkarım Günlüğü & İnceleme", "📈 Telemetri & Analiz", "🛠️ CRUD & İnsan Denetimi"])

# Model filtresini ayarla
filtre_model = None if model_secimi == "Tümü" else model_secimi
df_cikarimlar = db.cikarimlari_getir(limit=kayit_limiti, model_adi=filtre_model, min_guven=min_guven)

with tab1:
    st.subheader("📋 Kalıcı Çıkarım Logları")
    if df_cikarimlar.empty:
        st.info("Kriterlere uygun çıkarım kaydı bulunamadı. Kenar çubuğundan 'Sentetik Veri Ekle' butonuna basabilirsiniz.")
    else:
        st.dataframe(df_cikarimlar, use_container_width=True)

with tab2:
    st.subheader("📈 Sınıf Dağılımı ve Gecikme Grafikleri")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sınıf Frekans Dağılımı**")
        df_sinif = AITelemetriAnalizci.sinif_dagilimi_al(db)
        if not df_sinif.empty:
            st.bar_chart(data=df_sinif.set_index("sinif_adi")["adet"])
        else:
            st.info("Henüz tespit kaydı yok.")
    with c2:
        st.markdown("**Son İsteklerin Gecikme Trendi (ms)**")
        if not df_cikarimlar.empty:
            st.line_chart(data=df_cikarimlar["gecikme_ms"])
        else:
            st.info("Veri yok.")

with tab3:
    st.subheader("✍️ İnsan Denetimi (Human-in-the-Loop) & Güncelleme")
    secilen_istek = st.text_input("Geri Bildirim Verilecek İstek ID (Örn: trace_00001):")
    c_dogru, c_aciklama = st.columns([1, 3])
    with c_dogru:
        dogru_mu = st.radio("Model Tahmini Doğru mu?", ["Doğru (Onayla)", "Yanlış (Hata)"])
    with c_aciklama:
        aciklama = st.text_input("Denetçi Açıklaması / Düzeltme Notu:")

    if st.button("💾 Geri Bildirimi Kaydet"):
        if secilen_istek:
            basari = db.geri_bildirim_guncelle(
                istek_id=secilen_istek,
                dogru_mu=(dogru_mu == "Doğru (Onayla)"),
                aciklama=aciklama
            )
            if basari:
                st.success(f"'{secilen_istek}' için geri bildirim başarıyla güncellendi!")
            else:
                st.error("İstek ID bulunamadı.")
        else:
            st.warning("Lütfen bir İstek ID giriniz.")
