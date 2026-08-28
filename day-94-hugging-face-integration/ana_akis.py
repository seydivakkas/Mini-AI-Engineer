"""
Day 94: Hugging Face Model Hub Entegrasyonu, Konfigürasyon ve Model Paketleme
-----------------------------------------------------------------------------
Custom PreTrainedModel, PretrainedConfig, AutoClasses, SafeTensors ve Model Hub Paketleme.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import time
import random
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoModelForImageClassification, AutoConfig

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.hub_yoneticisi import HubPaketleyici
from src.gorsellestirici import HubGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 85)
    print("🚀 Day 94: Hugging Face Model Hub Entegrasyonu, Konfigürasyon ve Model Paketleme")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Özel MiniViT Konfigürasyonu ve Modelin İlklendirilmesi
    # -------------------------------------------------------------
    print("\n[1/4] MiniViTConfig ve MiniViTForImageClassification Oluşturuluyor...")
    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        giris_kanali=3,
        gizli_boyut=128,
        katman_sayisi=4,
        dikkat_baslik_sayisi=4,
        mlp_ara_boyut=256,
        dropout_orani=0.1,
        sinif_sayisi=10,
    )

    model = MiniViTForImageClassification(config).to(cihaz)
    toplam_parametre = sum(p.numel() for p in model.parameters())

    print(f"  ✓ Model Türü: `{config.model_type}`")
    print(f"  ✓ Toplam Parametre Sayısı: {toplam_parametre:,}")
    print(f"  ✓ Yama Sayısı (Patches): {(config.goruntu_boyutu // config.yama_boyutu)**2} (4x4 Yama Boyutu)")

    # -------------------------------------------------------------
    # ADIM 2: Hugging Face Standartlarında Paketleme ve SafeTensors Kaydı
    # -------------------------------------------------------------
    print("\n[2/4] Model SafeTensors Formatında Yerel Hub Dizinine Paketleniyor...")
    paketleyici = HubPaketleyici()
    hedef_dizin = os.path.join(os.path.dirname(__file__), "model_paketi")

    paket_bilgisi = paketleyici.modeli_paketle_ve_kaydet(
        model=model,
        kayit_dizini=hedef_dizin,
        repo_adi="seydivakkas/minivit-cifar10-v1",
    )

    print("=" * 85)
    print("📦 HUGGING FACE MODEL HUB PAKETİ İÇERİĞİ")
    print("=" * 85)
    for d_adi, boyut in paket_bilgisi.dosya_boyutlari_kb.items():
        print(f"  • {d_adi:<26}: {boyut:>8.2f} KB")

    print(f"\n• Toplam Paket Boyutu      : {sum(paket_bilgisi.dosya_boyutlari_kb.values()):.2f} KB")
    print(f"• Sayısal Doğrulama Farkı  : {paket_bilgisi.maks_hata_farki:.2e}")
    print(f"• Çıkarım Uyumluluğu       : {'✅ %100 BİREBİR EŞİT' if paket_bilgisi.sayisal_uyumluluk_dogrulandi else '❌ HATA'}")

    # -------------------------------------------------------------
    # ADIM 3: AutoModel Yükleme ve Çıkarım Gecikmesi Benchmark'ı
    # -------------------------------------------------------------
    print("\n[3/4] AutoModelForImageClassification ile Yükleme ve Benchmark Testi...")
    yuklenen_model = AutoModelForImageClassification.from_pretrained(hedef_dizin, local_files_only=True).to(cihaz)
    yuklenen_model.eval()

    cikarim_sureleri = []
    test_girdi = torch.randn(1, 3, config.goruntu_boyutu, config.goruntu_boyutu).to(cihaz)

    # Isınma (Warmup)
    for _ in range(5):
        with torch.no_grad():
            _ = yuklenen_model(test_girdi)

    # Ölçüm
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            cikti = yuklenen_model(test_girdi)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        cikarim_sureleri.append((t1 - t0) * 1000.0)  # ms

    ort_sure = float(np.mean(cikarim_sureleri))
    p95_sure = float(np.percentile(cikarim_sureleri, 95))
    print(f"  ✓ Ortalama Çıkarım Gecikmesi (P50): {ort_sure:.3f} ms")
    print(f"  ✓ P95 Çıkarım Gecikmesi            : {p95_sure:.3f} ms")

    # -------------------------------------------------------------
    # ADIM 4: 6-Panelli Görselleştirme Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6-Panelli Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = HubGorsellestirici()
    cikti_panosu = os.path.join(os.path.dirname(__file__), "ciktilar", "huggingface_entegrasyon_paneli.png")

    gorsellestirici.olustur_hf_entegrasyon_paneli(
        paket_bilgisi=paket_bilgisi,
        model=model,
        cikarim_sureleri=cikarim_sureleri,
        kayit_yolu=cikti_panosu,
    )
    print(f"  ✓ Teşhis Panosu Başarıyla Kaydedildi: {cikti_panosu}")
    print("\n✅ Day 94: Hugging Face Model Hub Entegrasyonu ve Paketleme Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
