"""
Day 90: GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Laboratuvarı
---------------------------------------------------------------------------------
Tekil Ardışık İstekler (B=1) vs Kuyruk Tabanlı Dinamik Batching Karşılaştırması,
GPU Tensör Çekirdeği Doygunluğu Analizi ve Gecikme Profili Çıkarımı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import time
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model import VisionClassifier
from src.dinamik_batcher import DinamikBatchMotoru
from src.benchmark_motoru import BatchingBenchmarkMotoru
from src.gorsellestirici import DinamikBatchGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def gpu_olcekleme_analizi(model: torch.nn.Module, cihaz: str = "cpu") -> dict:
    """Farklı batch boyutlarının GPU çıkarım sürelerine etkisini ölçer."""
    batch_boyutlari = [1, 2, 4, 8, 16, 32, 64]
    cikarim_sureleri = []

    model = model.to(cihaz).eval()

    # Isınma
    with torch.no_grad():
        _ = model(torch.randn(4, 3, 32, 32, device=cihaz))
        if cihaz == "cuda":
            torch.cuda.synchronize()

    for b in batch_boyutlari:
        x = torch.randn(b, 3, 32, 32, device=cihaz)
        tekrar = 30
        t0 = time.time()
        with torch.no_grad():
            for _ in range(tekrar):
                _ = model(x)
                if cihaz == "cuda":
                    torch.cuda.synchronize()
        ortalama_ms = ((time.time() - t0) / tekrar) * 1000.0
        cikarim_sureleri.append(ortalama_ms)

    return {
        "batch_boyutlari": batch_boyutlari,
        "cikarim_sureleri_ms": cikarim_sureleri
    }


def main():
    print("=" * 85)
    print("🚀 Day 90: GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Model ve Benchmark Motoru Hazırlığı
    model = VisionClassifier(giris_kanali=3, sinif_sayisi=10, taban_kanal=32)
    benchmark_motoru = BatchingBenchmarkMotoru(model=model, cihaz=cihaz)

    # 2. 200 Adet Sentetik İstemci İsteği Üret
    toplam_istek_sayisi = 200
    istek_tensörleri = [torch.randn(3, 32, 32) for _ in range(toplam_istek_sayisi)]

    # -------------------------------------------------------------
    # ADIM 1: GPU Alt-Doğrusal Ölçeklenme Analizi
    # -------------------------------------------------------------
    print("\n[1/3] GPU Tensör Çekirdeği Doygunluğu ve Alt-Doğrusal Ölçeklenme Ölçülüyor...")
    olcekleme_verisi = gpu_olcekleme_analizi(model, cihaz=cihaz)
    for b, t in zip(olcekleme_verisi["batch_boyutlari"], olcekleme_verisi["cikarim_sureleri_ms"]):
        print(f"  • Batch Boyutu: {b:>2} | Toplam Süre: {t:>6.2f} ms | Örnek Başına Maliyet: {t/b:>6.3f} ms/örnek")

    # -------------------------------------------------------------
    # ADIM 2: Tekil Ardışık Çıkarım (Sequential B=1)
    # -------------------------------------------------------------
    print(f"\n[2/3] Tekil Ardışık Çıkarım (Sequential B=1) Koşturuluyor ({toplam_istek_sayisi} İstek)...")
    ardisik_sonuc = benchmark_motoru.kos_ardisik_b1(istek_tensörleri)
    print(f"  ✓ Toplam Süre: {ardisik_sonuc['toplam_sure_sn']:.3f} s")
    print(f"  ✓ İşlem Hacmi (Throughput): {ardisik_sonuc['throughput_req_s']:.1f} req/s")
    print(f"  ✓ Ortalama Gecikme: {ardisik_sonuc['ortalama_gecikme_ms']:.2f} ms | P99: {ardisik_sonuc['p99_gecikme_ms']:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 3: Dinamik Batching Çıkarım Motoru (B_max=32, delay=8ms)
    # -------------------------------------------------------------
    print(f"\n[3/3] Dinamik Batching Çıkarım Motoru Koşturuluyor (16 Eşzamanlı İstemci)...")
    dinamik_sonuc = benchmark_motoru.kos_dinamik_batching(
        istekler=istek_tensörleri,
        max_batch_size=32,
        max_bekleme_ms=8.0,
        es_zamanli_istemci_sayisi=16
    )
    print(f"  ✓ Toplam Süre: {dinamik_sonuc['toplam_sure_sn']:.3f} s")
    print(f"  ✓ İşlem Hacmi (Throughput): {dinamik_sonuc['throughput_req_s']:.1f} req/s")
    print(f"  ✓ Ortalama Gecikme: {dinamik_sonuc['ortalama_gecikme_ms']:.2f} ms (Kuyruk: {dinamik_sonuc['ortalama_kuyruk_ms']:.2f} ms, Çıkarım: {dinamik_sonuc['ortalama_cikarim_ms']:.2f} ms)")
    print(f"  ✓ P50: {dinamik_sonuc['p50_gecikme_ms']:.2f} ms | P99: {dinamik_sonuc['p99_gecikme_ms']:.2f} ms")
    print(f"  ✓ Ortalama Oluşturulan Batch Boyutu: {dinamik_sonuc['ortalama_batch']:.1f}")

    hizlanma_orani = dinamik_sonuc["throughput_req_s"] / max(1e-5, ardisik_sonuc["throughput_req_s"])
    print(f"\n🔥 NET GPU HIZLANMA ÇARPANI: {hizlanma_orani:.2f}x DAHA YÜKSEK İŞLEM HACMİ!")

    # -------------------------------------------------------------
    # ADIM 4: Teşhis Panosu Görselleştirmesi
    # -------------------------------------------------------------
    gorsellestirici = DinamikBatchGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "dinamik_batching_paneli.png")

    gorsellestirici.olustur_batching_paneli(
        ardisik_sonuc=ardisik_sonuc,
        dinamik_sonuc=dinamik_sonuc,
        batch_olcekleme_verisi=olcekleme_verisi,
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 90: Dinamik Batching Çıkarım Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
