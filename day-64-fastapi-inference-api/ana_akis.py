"""
Day 64: FastAPI İnference, Model Lifespan & Batch Prediction Ana Yürütme Betiği.
"""

import os
import sys
import time
import asyncio
import httpx

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.api_servisi import olustur_uygulama
from src.batch_kuyruk_yoneticisi import DinamikBatchKuyrugu
from src.gorsellestirici import FastAPIGorsellestirici


async def calistir_benchmark():
    print("=" * 95, flush=True)
    print(">>> DAY 64: URETIM SEVIYESI FASTAPI INFERENCE, MODEL LIFESPAN & BATCH PREDICTION", flush=True)
    print("=" * 95, flush=True)

    app = olustur_uygulama()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Sağlık ve Lifespan Kontrolü
            print("\n[+] 1. Adim: /saglik Endpoint'i ile Model Lifespan Durumu Sorgulaniyor...", flush=True)
            saglik_res = await client.get("/saglik")
            print(f"    - HTTP Durumu      : {saglik_res.status_code}")
            print(f"    - Saglik Verisi    : {saglik_res.json()}")

            # 2. Tekil Çıkarım Benchmark Testi (Batch Size = 1)
            num_single_requests = 300
            print(f"\n[+] 2. Adim: {num_single_requests} Adet Tekil Cikarim (Batch=1) Test Ediliyor...", flush=True)

            start_t = time.perf_counter()
            for i in range(num_single_requests):
                payload = {
                    "istek_id": f"single_req_{i:04d}",
                    "gorsel_meta": {"genislik": 1920, "yukseklik": 1080, "format": "JPEG"},
                    "nms_esigi": 0.45,
                    "guven_esigi": 0.50
                }
                res = await client.post("/v1/tahmin/tekil", json=payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Hata: {res.text}")

            total_single_time = time.perf_counter() - start_t
            tekil_qps = float(num_single_requests / max(total_single_time, 1e-6))
            tekil_lat = float((total_single_time / num_single_requests) * 1000.0)
            print(f"    - Tekil QPS        : {tekil_qps:,.1f} QPS | Ortalama Gecikme: {tekil_lat:.2f} ms")

            # 3. Toplu Çıkarım Benchmark Testi (Batch Size = 32)
            batch_size = 32
            num_batches = 20
            total_batch_items = batch_size * num_batches
            print(f"\n[+] 3. Adim: {num_batches} Batch (Toplam {total_batch_items} Istek, B={batch_size}) Test Ediliyor...", flush=True)

            start_t = time.perf_counter()
            for b in range(num_batches):
                istekler = [
                    {
                        "istek_id": f"batch_req_{b:02d}_{i:02d}",
                        "gorsel_meta": {"genislik": 1920, "yukseklik": 1080, "format": "PNG"},
                        "nms_esigi": 0.45,
                        "guven_esigi": 0.50
                    }
                    for i in range(batch_size)
                ]
                batch_payload = {
                    "batch_id": f"batch_cluster_{b:03d}",
                    "istekler": istekler
                }
                res = await client.post("/v1/tahmin/toplu", json=batch_payload)
                if res.status_code != 200:
                    raise RuntimeError(f"Hata: {res.text}")

            total_batch_time = time.perf_counter() - start_t
            batch_qps = float(total_batch_items / max(total_batch_time, 1e-6))
            batch_lat = float((total_batch_time / total_batch_items) * 1000.0)
            hizlanma_orani = float(batch_qps / max(tekil_qps, 1e-6))
            print(f"    - Toplu QPS        : {batch_qps:,.1f} QPS | Istek Basi Amorti Gecikme: {batch_lat:.2f} ms")
            print(f"    - Hizlanma Kazanci : {hizlanma_orani:.2f}x")

            # 4. Asenkron Dinamik Batch Kuyruk Testi
            print("\n[+] 4. Adim: Asenkron Dinamik Batch Kuyrugu (Async Dynamic Queue) Test Ediliyor...", flush=True)
            motor = app.state.model_motoru
            kuyruk = DinamikBatchKuyrugu(model_motoru=motor, maks_batch_boyutu=16, maks_bekleme_ms=8.0)
            kuyruk.baslat()

            async def kuyruga_gonder(idx: int):
                req = {"istek_id": f"async_q_{idx:03d}", "nms_esigi": 0.45, "guven_esigi": 0.50}
                return await kuyruk.tahmin_kuyruga_ekle(req)

            gorevler = [kuyruga_gonder(i) for i in range(32)]
            kuyruk_sonuclari = await asyncio.gather(*gorevler)
            await kuyruk.durdur()
            print(f"    - Dinamik Kuyruk Islemi: {len(kuyruk_sonuclari)} Istek Basariyla Tamamlandi.")

        # 5. Özet ve Görselleştirme
        bench_data = {
            "tekil_qps": tekil_qps,
            "batch_qps": batch_qps,
            "tekil_ortalama_gecikme_ms": tekil_lat,
            "batch_istek_basi_gecikme_ms": batch_lat,
            "hizlanma_orani": hizlanma_orani
        }

        print("\n" + "=" * 95, flush=True)
        print(">>> 5. FASTAPI ASYNC INFERENCE PERFORMANS TABLOSU", flush=True)
        print("=" * 95, flush=True)
        print(f"* Tekil Cikarim Hizi (B=1)   : {tekil_qps:>10,.1f} QPS | Gecikme: {tekil_lat:>6.2f} ms")
        print(f"* Toplu Cikarim Hizi (B=32)  : {batch_qps:>10,.1f} QPS | Gecikme: {batch_lat:>6.2f} ms")
        print(f"* Batching Hizlanma Carpani  : {hizlanma_orani:>10.2f}x Daha Yuksek Throughput")
        print(f"* Model Lifespan Durumu      : %100 Hazir (app.state baglantisi basarili)")

        hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "fastapi_inference_paneli.png")
        cikis = FastAPIGorsellestirici.panel_ciz(bench_data, hedef_path=hedef_pano)
        print(f"\n[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis)}", flush=True)
        print("=" * 95, flush=True)
        print("DAY 64: FASTAPI INFERENCE & MODEL LIFESPAN BASARIYLA TAMAMLANDI!", flush=True)
        print("=" * 95, flush=True)


def main():
    asyncio.run(calistir_benchmark())


if __name__ == "__main__":
    main()
