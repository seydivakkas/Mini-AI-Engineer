"""
Day 36: Streamlit ile İnteraktif AI Kontrol Paneli Birim Testleri.
"""

import os
import pytest
import numpy as np
from PIL import Image
from src.ai_modulleri import DashboardAIEngine
from src.bilesenler import tespit_kutularini_ciz
from src.gorsellestirici import StreamlitDashboardGorsellestirici


@pytest.fixture
def ai_engine():
    return DashboardAIEngine(device="cpu")


def test_ai_engine_text_analysis(ai_engine):
    """Metin sınıflandırma ve olasılık toplamı testi."""
    metin = "Evrisimli sinir aglari ile gorsel siniflandirma"
    res = ai_engine.metin_analiz_et(metin)
    assert "tahmin_sinifi" in res
    assert 0.0 <= res["guven"] <= 1.0
    assert len(res["embedding"]) == 64
    assert sum(res["olasiliklar"].values()) == pytest.approx(1.0, 1e-4)


def test_ai_engine_image_analysis(ai_engine):
    """Görsel kusur tespiti ve eşikleme testi."""
    sentetik = np.ones((200, 200, 3), dtype=np.uint8) * 128
    img = Image.fromarray(sentetik)

    res_all = ai_engine.gorsel_analiz_et(img, guven_esigi=0.1)
    res_high = ai_engine.gorsel_analiz_et(img, guven_esigi=0.8)

    assert len(res_all["tespitler"]) >= len(res_high["tespitler"])
    assert res_all["genislik"] == 200
    assert res_all["yukseklik"] == 200
    assert len(res_all["baskin_renk"]) == 3


def test_ai_engine_rag_qa(ai_engine):
    """RAG doküman soru-cevap ve atıf testi."""
    soru = "YOLO nasil calisir?"
    res = ai_engine.rag_soru_sor(soru, top_k=2)
    assert len(res["kaynaklar"]) >= 1
    assert "KB-01" in res["kaynaklar"] or "KB-02" in res["kaynaklar"]
    assert res["guven"] >= 0.5


def test_bilesenler_tespit_cizimi():
    """PIL üzerine bounding box çizim fonksiyonu testi."""
    img = Image.new("RGB", (300, 300), color="white")
    tespitler = [
        {"etiket": "Kusur_A", "guven": 0.95, "kutu": [20, 20, 100, 100]}
    ]
    cizili = tespit_kutularini_ciz(img, tespitler)
    assert cizili.size == (300, 300)


def test_ai_engine_latency_telemetry(ai_engine):
    """Telemetri ve istek sayaçları testi."""
    ai_engine.metin_analiz_et("Test 1")
    ai_engine.metin_analiz_et("Test 2")
    assert ai_engine.istek_sayisi == 2
    assert len(ai_engine.gecikme_gecmisi) == 2


def test_dashboard_gorsellestirici(tmp_path):
    """6 panelli dashboard görselleştirici testi."""
    metin_mock = {
        "olasiliklar": {"A": 0.6, "B": 0.4}
    }
    gorsel_mock = {
        "tespitler": [{"etiket": "Hata", "guven": 0.9, "kutu": [10, 10, 50, 50]}]
    }
    cikis_path = str(tmp_path / "test_dash.png")
    yol = StreamlitDashboardGorsellestirici.dashboard_paneli_ciz(metin_mock, gorsel_mock, hedef_path=cikis_path)
    assert os.path.exists(yol)
