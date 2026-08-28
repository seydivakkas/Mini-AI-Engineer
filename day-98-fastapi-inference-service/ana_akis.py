"""
Day 98: MiniViT v1.0 FastAPI Asenkron Çıkarım Servisi Ana Akış ve Performans Benchmark'ı.
"""

import os
import sys
import io
import time
import base64
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.api_uygulamasi import app
from src.gorsellestirici import FastAPIGorsellestirici


def main():
    print("=" * 85)
    print(">>> Day 98: MiniViT v1.0 Üretime Hazır Yüksek Performanslı Asenkron FastAPI Servisi")
    print("=" * 85)

    with TestClient(app) as client:
        # -------------------------------------------------------------
        # ADIM 1: Kubernetes Liveness & Readiness Sağlık Kontrolleri
        # -------------------------------------------------------------
        print("\n[1/5] Kubernetes Sağlık Probeları (/health, /ready) Test Ediliyor...")
        saglik_yaniti = client.get("/health").json()
        metaveri_yaniti = client.get("/metadata").json()

        print("=" * 85)
        print("[-] SERVIS SAGLIK VE METAVERI RAPORU")
        print("=" * 85)
        print(f"  * Servis Durumu          : {saglik_yaniti['status']} [OK]")
        print(f"  * Model Yüklenme Durumu  : {saglik_yaniti['model_loaded']}")
        print(f"  * Çıkarım Cihazı         : {saglik_yaniti['cihaz'].upper()}")
        print(f"  * Model Adı              : {metaveri_yaniti['model_adi']}")
        print(f"  * Toplam Parametre       : {metaveri_yaniti['parametre_sayisi']:,}")
        print(f"  * Sınıf Sayısı           : {metaveri_yaniti['sinif_sayisi']}")

        # -------------------------------------------------------------
        # ADIM 2: Tekli Görsel Çıkarım Testi (Multipart Form-Data)
        # -------------------------------------------------------------
        print("\n[2/5] Tekli Görsel Çıkarım Endpoint'i (/predict) Test Ediliyor...")
        ornek_img = Image.new("RGB", (32, 32), color=(50, 100, 200))
        img_buf = io.BytesIO()
        ornek_img.save(img_buf, format="PNG")
        img_bytes = img_buf.getvalue()

        # Isınma
        for _ in range(5):
            _ = client.post("/predict?top_k=5", files={"file": ("test.png", img_bytes, "image/png")})

        ornek_tahmin_res = client.post("/predict?top_k=5", files={"file": ("test.png", img_bytes, "image/png")}).json()

        print("=" * 85)
        print("[-] CANLI TAHMIN SONUCLARI (TOP-5)")
        print("=" * 85)
        print(f"  * En İyi Tahmin : {ornek_tahmin_res['en_iyi_tahmin']['sinif_adi']} "
              f"(%{ornek_tahmin_res['en_iyi_tahmin']['olasilik']*100:.2f})")
        print(f"  * Servis Gecikmesi: {ornek_tahmin_res['gecikme_ms']:.2f} ms")
        print("  * Top-5 Sıralaması:")
        for t in ornek_tahmin_res["top_k_tahminler"]:
            print(f"    - {t['sinif_adi']:<12}: %{t['olasilik']*100:>6.2f}")

        # -------------------------------------------------------------
        # ADIM 3: Base64 ve Toplu (Batch) Çıkarım Testleri
        # -------------------------------------------------------------
        print("\n[3/5] Base64 ve Toplu Çıkarım (/predict/base64 & /predict/batch) Test Ediliyor...")
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        b64_res = client.post("/predict/base64", json={"base64_goruntu": b64_str, "top_k": 3}).json()
        print(f"  * Base64 İstek Gecikmesi: {b64_res['gecikme_ms']:.2f} ms")

        batch_files = [
            ("files", ("res1.png", img_bytes, "image/png")),
            ("files", ("res2.png", img_bytes, "image/png")),
            ("files", ("res3.png", img_bytes, "image/png")),
        ]
        batch_res = client.post("/predict/batch?top_k=3", files=batch_files).json()
        print(f"  * Batch (3 Görsel) Toplam Gecikme: {batch_res['toplam_gecikme_ms']:.2f} ms")

        # -------------------------------------------------------------
        # ADIM 4: Gecikme Benchmark'ı ve Metrik Toplama
        # -------------------------------------------------------------
        print("\n[4/5] 50 İsteklik Gecikme Benchmark'ı Yapılıyor...")
        gecikmeler = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = client.post("/predict?top_k=5", files={"file": ("test.png", img_bytes, "image/png")})
            t1 = time.perf_counter()
            gecikmeler.append((t1 - t0) * 1000.0)

        metrik_verisi = client.get("/metrics").json()
        p50 = float(np.percentile(gecikmeler, 50))
        p90 = float(np.percentile(gecikmeler, 90))
        p99 = float(np.percentile(gecikmeler, 99))

        print("=" * 85)
        print("[-] FASTAPI GECIKME PERFORMANS RAPORU (50 ISTEK)")
        print("=" * 85)
        print(f"  * P50 Medyan Gecikme   : {p50:.2f} ms")
        print(f"  * P90 Gecikme          : {p90:.2f} ms")
        print(f"  * P99 Gecikme          : {p99:.2f} ms")
        print(f"  * Toplam Karşılanan    : {metrik_verisi['toplam_istek_sayisi']} İstek")

        # -------------------------------------------------------------
        # ADIM 5: 6 Panelli Teşhis Panosu Oluşturma
        # -------------------------------------------------------------
        print("\n[5/5] 6 Panelli Teşhis Panosu Çiziliyor...")
        gorsellestirici = FastAPIGorsellestirici(dpi=300)
        cikis_resmi = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ciktilar",
            "fastapi_servis_paneli.png",
        )
        gorsellestirici.pano_olustur(
            metrik_verisi=metrik_verisi,
            gecikmeler=gecikmeler,
            ornek_tahminler=ornek_tahmin_res["top_k_tahminler"],
            saglik_verisi=saglik_yaniti,
            kayit_yolu=cikis_resmi,
        )

    print("\n" + "=" * 85)
    print("[OK] Day 98: FastAPI Asenkron Cikarim Servisi ve Health Probelari Basariyla Tamamlandi!")
    print("=" * 85)


if __name__ == "__main__":
    main()
