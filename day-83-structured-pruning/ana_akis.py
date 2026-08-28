"""
Day 83: L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama Laboratuvarı
------------------------------------------------------------------
Filtre önem skorlama, fiziksel katman küçültme (Layer Stitching), çıkarım gecikmesi (ms)
ve ince ayar (Fine-Tuning) ile doğruluk toparlama akışı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model import BudanabilirVisionCNN
from src.budayici import YapisalFiltreBudayici
from src.olcumleyici import PerformansOlcumleyici
from src.gorsellestirici import BudamaGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def veri_olustur(ornek_sayisi: int = 600, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 0.8
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        x[i, :, (c%4)*7:(c%4)*7+6, (c//4)*7:(c//4)*7+6] += 1.5
    return x, y


def main():
    print("=" * 85)
    print("🚀 Day 83: L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    tr_x, tr_y = veri_olustur(640, 10)
    val_x, val_y = veri_olustur(160, 10)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)

    # 2. Orijinal Yoğun (Dense) Modelin Eğitimi
    print(f"\n[1/4] Orijinal Yoğun Model (Dense [32, 64, 128]) Eğitiliyor...")
    model_dense = BudanabilirVisionCNN(kanallar=[32, 64, 128]).to(cihaz)
    opt = torch.optim.AdamW(model_dense.parameters(), lr=2e-3, weight_decay=1e-4)

    for _ in range(8):
        model_dense.train()
        for bx, by in tr_loader:
            bx, by = bx.to(cihaz), by.to(cihaz)
            opt.zero_grad()
            loss = F.cross_entropy(model_dense(bx), by)
            loss.backward()
            opt.step()

    acc_dense = PerformansOlcumleyici.dogruluk_olc(model_dense, val_loader, cihaz)
    params_dense = PerformansOlcumleyici.parametre_sayisi(model_dense)
    gecikme_dense = PerformansOlcumleyici.cikarim_gecikmesi_ve_fps(model_dense, (32, 3, 32, 32), cihaz)["gecikme_ms"]

    print(f"  ✓ Yoğun Model Parametre: {params_dense:,}")
    print(f"  ✓ Yoğun Model Doğruluk: %{acc_dense:.2f}")
    print(f"  ✓ Yoğun Model Gecikme: {gecikme_dense:.2f} ms")

    # 3. Deney 1: %25 Yapısal Budama
    print(f"\n[2/4] Deney 1: %25 L1-Norm Yapısal Filtre Budama Uygulanıyor...")
    model_p25, rapor_p25 = YapisalFiltreBudayici.modeli_yapisal_buda(model_dense, budama_orani=0.25, norm_tipi="L1")
    acc_p25_once = PerformansOlcumleyici.dogruluk_olc(model_p25, val_loader, cihaz)
    params_p25 = PerformansOlcumleyici.parametre_sayisi(model_p25)

    _ = PerformansOlcumleyici.ince_ayar_yap(model_p25, tr_loader, val_loader, epok_sayisi=4, lr=3e-4, cihaz=cihaz)
    acc_p25_sonra = PerformansOlcumleyici.dogruluk_olc(model_p25, val_loader, cihaz)
    gecikme_p25 = PerformansOlcumleyici.cikarim_gecikmesi_ve_fps(model_p25, (32, 3, 32, 32), cihaz)["gecikme_ms"]

    print(f"  ✓ %25 Budanmış Kanallar: {rapor_p25['yeni_kanallar']}")
    print(f"  ✓ Parametre: {params_p25:,} (Tasarrruf: %{(1 - params_p25/params_dense)*100:.1f})")
    print(f"  ✓ Budama Hemen Sonrası Doğruluk: %{acc_p25_once:.2f} ──> Fine-Tuning Sonrası: %{acc_p25_sonra:.2f}")
    print(f"  ✓ Yeni Gecikme: {gecikme_p25:.2f} ms (Hızlanma: %{(1 - gecikme_p25/gecikme_dense)*100:.1f})")

    # 4. Deney 2: %50 Yapısal Budama
    print(f"\n[3/4] Deney 2: %50 L1-Norm Yapısal Filtre Budama Uygulanıyor...")
    model_p50, rapor_p50 = YapisalFiltreBudayici.modeli_yapisal_buda(model_dense, budama_orani=0.50, norm_tipi="L1")
    acc_p50_once = PerformansOlcumleyici.dogruluk_olc(model_p50, val_loader, cihaz)
    params_p50 = PerformansOlcumleyici.parametre_sayisi(model_p50)

    _ = PerformansOlcumleyici.ince_ayar_yap(model_p50, tr_loader, val_loader, epok_sayisi=4, lr=3e-4, cihaz=cihaz)
    acc_p50_sonra = PerformansOlcumleyici.dogruluk_olc(model_p50, val_loader, cihaz)
    gecikme_p50 = PerformansOlcumleyici.cikarim_gecikmesi_ve_fps(model_p50, (32, 3, 32, 32), cihaz)["gecikme_ms"]

    print(f"  ✓ %50 Budanmış Kanallar: {rapor_p50['yeni_kanallar']}")
    print(f"  ✓ Parametre: {params_p50:,} (Tasarrruf: %{(1 - params_p50/params_dense)*100:.1f})")
    print(f"  ✓ Budama Hemen Sonrası Doğruluk: %{acc_p50_once:.2f} ──> Fine-Tuning Sonrası: %{acc_p50_sonra:.2f}")
    print(f"  ✓ Yeni Gecikme: {gecikme_p50:.2f} ms (Hızlanma: %{(1 - gecikme_p50/gecikme_dense)*100:.1f})")

    # 5. Görselleştirme
    print(f"\n[4/4] 6 Panelli Teşhis Panosu Oluşturuluyor...")
    skorlar_conv2 = rapor_p25["katman_skorlari"]["conv2"]
    esik_degeri = np.percentile(skorlar_conv2, 25)

    gorsellestirici = BudamaGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "structured_pruning_paneli.png")

    gorsellestirici.olustur_budama_paneli(
        katman_skorlari=skorlar_conv2,
        budama_esigi=esik_degeri,
        oranlar=["%0 (Yoğun)", "%25 Budanmış", "%50 Budanmış"],
        parametreler=[params_dense, params_p25, params_p50],
        gecikmeler=[gecikme_dense, gecikme_p25, gecikme_p50],
        dogruluk_oncesi=[acc_dense, acc_p25_once, acc_p50_once],
        dogruluk_sonrasi=[acc_dense, acc_p25_sonra, acc_p50_sonra],
        kayit_yolu=cikti_yolu
    )

    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 83: L1/L2 Norm Structured Pruning Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
