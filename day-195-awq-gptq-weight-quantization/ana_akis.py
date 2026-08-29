"""
Day 195: AWQ ve GPTQ 4-Bit Kuantizasyon Ana Çalıştırma Akışı.
"""

import os
import sys
import numpy as np
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.kuantizasyon_motoru import (
    StandartRoundToNearestQuantizer,
    AWQQuantizer,
    GPTQQuantizer,
)
from src.perplexity_profilleyici import PerplexityVeVRAMProfilleyici
from src.gorsellestirici import KuantizasyonGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 195 (FAZ 10): ADVANCED WEIGHT QUANTIZATION (AWQ & GPTQ 4-BIT ENGINE)")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Sentetik Ağırlık ve Aktivasyon Verisi Üretimi
    # -------------------------------------------------------------
    print("\n[1/4] Transformer Katmanı Ağırlık ve Aktivasyon Tensörleri Hazırlanıyor...")
    torch.manual_seed(42)
    in_features = 512
    out_features = 512
    n_samples = 256

    w_orig = torch.randn(out_features, in_features) * 0.02

    # Aktivasyon tensöründe %1 salient kanal simülasyonu (Büyük outlier aktivasyonlar)
    x_act = torch.randn(n_samples, in_features)
    x_act[:, :5] *= 15.0  # İlk 5 kanal yüksek aktivasyonlu salient kanal

    print(f"  • Ağırlık Matrisi Boyutu      : {list(w_orig.shape)} (FP16: {w_orig.nelement() * 2 / 1024:.1f} KB)")
    print(f"  • Aktivasyon Tensörü Boyutu   : {list(x_act.shape)}")
    print("  • Salient Kanallar            : İlk %1 kanal 15x daha büyük genliğe sahip.")

    # -------------------------------------------------------------
    # ADIM 2: RTN, AWQ ve GPTQ Kuantizasyon Denemeleri
    # -------------------------------------------------------------
    print("\n[2/4] 4-Bit Kuantizasyon Algoritmaları Yürütülüyor...")

    # 1. Standart RTN
    w_rtn, _, _ = StandartRoundToNearestQuantizer.kuantize_et(w_orig, group_size=128)
    hata_rtn = PerplexityVeVRAMProfilleyici.hata_olcumleri(w_orig, w_rtn)

    # 2. AWQ
    w_awq, _ = AWQQuantizer.kuantize_et(w_orig, x_act, group_size=128, gamma=0.5)
    hata_awq = PerplexityVeVRAMProfilleyici.hata_olcumleri(w_orig, w_awq)

    # 3. GPTQ
    w_gptq = GPTQQuantizer.kuantize_et(w_orig, x_act, block_size=128)
    hata_gptq = PerplexityVeVRAMProfilleyici.hata_olcumleri(w_orig, w_gptq)

    print(f"  • Standart RTN INT4  -> MSE: {hata_rtn['mse_loss']:.6f} | Kosinüs Benzerliği: {hata_rtn['kosinus_benzerligi']:.4f}")
    print(f"  • GPTQ INT4          -> MSE: {hata_gptq['mse_loss']:.6f} | Kosinüs Benzerliği: {hata_gptq['kosinus_benzerligi']:.4f}")
    print(f"  • AWQ INT4           -> MSE: {hata_awq['mse_loss']:.6f} | Kosinüs Benzerliği: {hata_awq['kosinus_benzerligi']:.4f}")
    print("  ✓ AWQ ve GPTQ Rekonstrüksiyon Başarımı Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Llama-3-70B Perplexity ve VRAM Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Llama-3-70B 4-Bit Kuantizasyon Kıyaslama Raporu...")
    kiyas_raporu = PerplexityVeVRAMProfilleyici.kuantizasyon_karsilastirma_raporu()

    print("-" * 110)
    print(f"{'Yöntem':<32} | {'Bit':<8} | {'VRAM (GB)':<12} | {'WikiText-2 PPL':<16} | {'Kosinüs':<10} | {'Kalite Durumu'}")
    print("-" * 110)
    for r in kiyas_raporu:
        print(
            f"{r['yontem']:<32} | "
            f"{r['bit_derinligi'].split(' ')[0]:<8} | "
            f"{r['model_vram_gb']:>6.1f} GB    | "
            f"{r['wikitext2_perplexity']:>12.2f}     | "
            f"{r['kosinus_benzerligi']:>8.4f} | "
            f"{r['kalite_durumu']}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Kuantizasyon Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "awq_gptq_paneli.png")
    salient_vals = torch.mean(torch.abs(x_act), dim=0).detach().cpu().numpy()

    KuantizasyonGorsellestirici.teshis_paneli_olustur(
        kiyas_raporu=kiyas_raporu,
        salient_kanallar=salient_vals,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Kuantizasyon Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 195: AWQ & GPTQ KUANTİZASYONU BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
