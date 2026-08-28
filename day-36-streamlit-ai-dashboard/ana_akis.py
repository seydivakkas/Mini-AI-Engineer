"""
Day 36: Streamlit ile İnteraktif AI Kontrol Paneli Ana Yürütme Betiği.
"""

import os
import numpy as np
from PIL import Image
from src.ai_modulleri import DashboardAIEngine
from src.bilesenler import tespit_kutularini_ciz
from src.gorsellestirici import StreamlitDashboardGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Streamlit Dashboard AI Motorunun Başlatılması")
    print("=" * 80)

    engine = DashboardAIEngine(device="cpu")

    # 1. Metin Sınıflandırma Analizi
    metin = "Evrişimli sinir ağları ve YOLO mimarisi ile görsel nesne tespiti"
    text_res = engine.metin_analiz_et(metin)
    print(f"[+] Metin Analizi:")
    print(f"    - Girdi: '{metin}'")
    print(f"    - Tahmin: {text_res['tahmin_sinifi']} (Güven: %{text_res['guven']*100:.1f})")
    print(f"    - Gecikme: {text_res['gecikme_ms']:.2f} ms")

    # 2. Sentetik Kumaş Görseli ve Kusur Tespiti
    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Görsel Kusur Tespiti ve Bounding Box Çizimi")
    print("=" * 80)
    sentetik_kumas = np.ones((300, 400, 3), dtype=np.uint8) * 190
    img = Image.fromarray(sentetik_kumas)

    gorsel_res = engine.gorsel_analiz_et(img, guven_esigi=0.5)
    cizili_img = tespit_kutularini_ciz(img, gorsel_res["tespitler"])

    print(f"[+] Görsel Analizi:")
    print(f"    - Boyut: {gorsel_res['genislik']}x{gorsel_res['yukseklik']}")
    print(f"    - Tespit Edilen Kusur: {len(gorsel_res['tespitler'])} adet")
    print(f"    - Baskın RGB: {gorsel_res['baskin_renk']}")
    print(f"    - Gecikme: {gorsel_res['gecikme_ms']:.2f} ms")

    # 3. RAG Soru-Cevap
    print("\n" + "=" * 80)
    print(">>> ASAMA 3: RAG Doküman Asistanı Soru-Cevap")
    print("=" * 80)
    soru = "YOLO nesne tespiti nasıl çalışır?"
    rag_res = engine.rag_soru_sor(soru, top_k=2)
    print(f"[+] RAG Analizi:")
    print(f"    - Soru: '{rag_res['soru']}'")
    print(f"    - Yanıt: '{rag_res['yanit']}'")
    print(f"    - Kaynaklar: {rag_res['kaynaklar']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: 6 Panelli Streamlit Dashboard Teşhis Panosunun Üretilmesi")
    print("=" * 80)
    cikis_resmi = StreamlitDashboardGorsellestirici.dashboard_paneli_ciz(
        ornek_metin_sonuc=text_res,
        ornek_gorsel_sonuc=gorsel_res,
        hedef_path="day-36-streamlit-ai-dashboard/ciktilar/streamlit_dashboard_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 36: STREAMLIT İLE İNTERAKTİF AI KONTROL PANELİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
