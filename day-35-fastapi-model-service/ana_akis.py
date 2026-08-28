"""
Day 35: FastAPI ile Asenkron AI Model Servisi & REST API Ana Yürütme Betiği.
"""

import os
import io
from fastapi.testclient import TestClient
from src.servis_uygulamasi import app
from src.gorsellestirici import FastAPIServisGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: FastAPI Test İstemcisinin Başlatılması ve Lifespan Kontrolü")
    print("=" * 80)

    client = TestClient(app)

    # 1. Sağlık Kontrolü (Healthz)
    resp_health = client.get("/healthz")
    print(f"[+] GET /healthz -> Durum Kodu: {resp_health.status_code}")
    print(f"    - Yanıt: {resp_health.json()}")
    print(f"    - X-Process-Time-Ms Header: {resp_health.headers.get('X-Process-Time-Ms')} ms")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Çoklu Modalite Uç Noktalarının Test Edilmesi")
    print("=" * 80)

    # 2. Metin Tahmini (Text Prediction)
    metin_istegi = {
        "metin": "Transformer ve Dikkat Mekanizmaları ile Dil Modeli Eğitimi",
        "kategori": "NLP",
        "embedding_iste": True
    }
    resp_text = client.post("/api/v1/predict/text", json=metin_istegi)
    text_data = resp_text.json()
    print(f"[+] POST /api/v1/predict/text -> Durum Kodu: {resp_text.status_code}")
    print(f"    - Tahmin: {text_data['tahmin_edilen_etiket']} (Güven: %{text_data['olasilik']*100:.1f})")
    print(f"    - Embedding Boyutu: {len(text_data.get('vektor_embedding') or [])} boyut")
    print(f"    - Model Gecikmesi : {text_data['gecikme_ms']:.2f} ms")

    # 3. Görsel Analizi (Image Prediction)
    sahte_resim_baytlari = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff\x00\xaa" * 50
    files = {"dosya": ("ornek_kumas.png", io.BytesIO(sahte_resim_baytlari), "image/png")}
    resp_img = client.post("/api/v1/predict/image", files=files)
    img_data = resp_img.json()
    print(f"\n[+] POST /api/v1/predict/image -> Durum Kodu: {resp_img.status_code}")
    print(f"    - Dosya Adı: {img_data['dosya_adi']} ({img_data['boyut_bayt']} bayt)")
    print(f"    - Tespit Sayısı: {len(img_data['tespit_edilen_nesneler'])} nesne")
    print(f"    - Baskın RGB: {img_data['en_baskin_renk']}")

    # 4. RAG Soru-Cevap (RAG Query)
    rag_istegi = {"soru": "FastAPI ile asenkron REST API nasıl kurulur?", "top_k": 2}
    resp_rag = client.post("/api/v1/rag/query", json=rag_istegi)
    rag_data = resp_rag.json()
    print(f"\n[+] POST /api/v1/rag/query -> Durum Kodu: {resp_rag.status_code}")
    print(f"    - Soru    : {rag_data['soru']}")
    print(f"    - Yanıt   : {rag_data['yanit']}")
    print(f"    - Kaynaklar: {rag_data['kaynaklar']}")

    # 5. Doğrulama Hatası Testi (Validation Error Handling)
    gecersiz_istek = {"metin": "a"}  # min_length=3 kuralı ihlal edilir
    resp_hata = client.post("/api/v1/predict/text", json=gecersiz_istek)
    print(f"\n[+] POST /api/v1/predict/text (Geçersiz İstek) -> Durum Kodu: {resp_hata.status_code}")
    print(f"    - Hata Yanıtı: {resp_hata.json()}")

    # 6. Telemetri Kontrolü (Telemetry API)
    resp_telem = client.get("/api/v1/telemetry")
    telem_data = resp_telem.json()
    print(f"\n[+] GET /api/v1/telemetry -> Toplam İşlenen İstek: {telem_data['toplam_istek']}")
    print(f"    - Ortalama Gecikme: {telem_data['ortalama_gecikme_ms']:.2f} ms")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: 6 Panelli FastAPI Teşhis Panosunun Üretilmesi")
    print("=" * 80)
    cikis_resmi = FastAPIServisGorsellestirici.servis_paneli_ciz(
        telemetri_verisi=telem_data,
        ornek_tahmin={
            "tum_olasiliklar": text_data.get("tum_olasiliklar", {}),
            "baskin_renk": img_data.get("en_baskin_renk", [52, 152, 219])
        },
        hedef_path="day-35-fastapi-model-service/ciktilar/fastapi_servis_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 35: FASTAPI ASENKRON AI MODEL SERVİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
