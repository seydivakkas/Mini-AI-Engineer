"""
Streamlit Yeniden Kullanılabilir Arayüz Bileşenleri (Reusable UI Components).
"""

from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import streamlit as st


def metrik_kartlari_ciz(toplam_istek: int, ortalama_gecikme: float, aktif_model: str):
    """Üst bilgi metrik kartlarını çizer."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="📊 Toplam Çıkarım İsteği", value=f"{toplam_istek:,}")
    with col2:
        st.metric(label="⚡ Ort. Model Gecikmesi", value=f"{ortalama_gecikme:.2f} ms")
    with col3:
        st.metric(label="🤖 Aktif AI Modeli", value=aktif_model)


def tespit_kutularini_ciz(orijinal_gorsel: Image.Image, tespitler: List[Dict[str, Any]]) -> Image.Image:
    """Görsel üzerine sınırlayıcı kutuları ve etiketleri çizer."""
    cizim_gorseli = orijinal_gorsel.copy()
    draw = ImageDraw.Draw(cizim_gorseli)

    renk_paleti = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

    for i, t in enumerate(tespitler):
        kutu = t["kutu"]  # [x1, y1, x2, y2]
        etiket = t["etiket"]
        guven = t["guven"]
        renk = renk_paleti[i % len(renk_paleti)]

        # Kutu çizgisi
        draw.rectangle(kutu, outline=renk, width=3)

        # Etiket arka planı ve metin
        metin = f"{etiket} %{guven*100:.1f}"
        draw.rectangle([kutu[0], max(0, kutu[1] - 18), kutu[0] + len(metin)*8 + 6, kutu[1]], fill=renk)
        draw.text((kutu[0] + 3, max(0, kutu[1] - 16)), metin, fill="white")

    return cizim_gorseli


def sohbet_mesajlarini_ciz(sohbet_gecmisi: List[Dict[str, str]]):
    """RAG sohbet geçmişini Streamlit sohbet balonları ile render eder."""
    for mesaj in sohbet_gecmisi:
        rol = mesaj.get("rol", "user")
        with st.chat_message(rol):
            st.write(mesaj.get("icerik", ""))
            if "kaynaklar" in mesaj and mesaj["kaynaklar"]:
                st.caption(f"📚 Kaynak Referansları: {', '.join(mesaj['kaynaklar'])}")


def guven_cubugu_ciz(olasiliklar: Dict[str, float]):
    """Sınıf olasılıklarını renkli ilerleme çubukları ile gösterir."""
    for sinif_adi, olasilik in olasiliklar.items():
        st.write(f"**{sinif_adi}** (%{olasilik*100:.1f})")
        st.progress(float(olasilik))
