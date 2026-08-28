"""
Day 84: Olasılık Kalibrasyonu, ECE ve Temperature Scaling Laboratuvarı
---------------------------------------------------------------------
Aşırı güven (Overconfidence) tespiti, Expected Calibration Error (ECE) ölçümü,
L-BFGS ile Temperature Scaling kalibrasyonu ve güvenilirlik diyagramları.

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

from src.model import GuvenilmezVisionModeli
from src.metrikler import KalibrasyonMetrikleri
from src.kalibrator import SicaklikKalibratoru
from src.gorsellestirici import KalibrasyonGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def veri_olustur(ornek_sayisi: int = 800, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    """
    Sınıf sınırlarında örtüşme ve belirsizlik içeren, aşırı güven (overconfidence)
    davranışını net ortaya koyan sentetik veri kümesi.
    """
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 1.2
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        # Ana sınıf sinyali (orta güçte)
        x[i, :, (c%4)*7:(c%4)*7+5, (c//4)*7:(c//4)*7+5] += 1.0
        # Yanıltıcı gürültü sinyali
        c_noise = (c + 2) % sinif_sayisi
        x[i, :, (c_noise%4)*7:(c_noise%4)*7+5, (c_noise//4)*7:(c_noise//4)*7+5] += 0.5
    return x, y


def main():
    print("=" * 85)
    print("🚀 Day 84: Olasılık Kalibrasyonu, ECE ve Temperature Scaling Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Kümeleri Hazırlığı (Train / Val / Test)
    tr_x, tr_y = veri_olustur(800, 10)
    val_x, val_y = veri_olustur(200, 10)
    test_x, test_y = veri_olustur(200, 10)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    # 2. Modelin Eğitilmesi
    print(f"\n[1/4] Derin Model Eğitiliyor (Aşırı Güvenli Rejime Sokuluyor)...")
    model = GuvenilmezVisionModeli(sinif_sayisi=10, taban_kanal=32).to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    for _ in range(12):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(cihaz), by.to(cihaz)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()

    # 3. Logitlerin Toplanması ve Ham Model ECE Değerlendirmesi
    print(f"\n[2/4] Doğrulama ve Test Logitleri Toplanıyor & Kalibrasyon Öncesi Ölçülüyor...")
    val_logitler, val_etiketler = GuvenilmezVisionModeli.logit_topla(model, val_loader, cihaz)
    test_logitler, test_etiketler = GuvenilmezVisionModeli.logit_topla(model, test_loader, cihaz)

    metrikler_ham = KalibrasyonMetrikleri.hesapla_tum_metrikler(test_logitler, test_etiketler, n_bins=15)
    print(f"  ✓ Ham Model Test Doğruluğu: %{metrikler_ham['dogruluk']:.2f}")
    print(f"  ✓ Ham Model Test ECE: %{metrikler_ham['ece']:.2f} (Beklenen Kalibrasyon Hatası)")
    print(f"  ✓ Ham Model Test NLL: {metrikler_ham['nll']:.4f}")
    print(f"  ✓ Ham Model Brier Skoru: {metrikler_ham['brier_score']:.4f}")

    # 4. Temperature Scaling ile Kalibrasyon (Val kümesi üzerinde)
    print(f"\n[3/4] Val Kümesi Üzerinde Temperature Scaling (L-BFGS) Optimize Ediliyor...")
    kalibrator = SicaklikKalibratoru(baslangic_sicaklik=1.5)
    rapor_kalibrasyon = kalibrator.kalibre_et(val_logitler, val_etiketler)
    optimal_t = rapor_kalibrasyon["optimal_sicaklik"]
    print(f"  ✓ Bulunan Optimal Sıcaklık Parametresi (T*): {optimal_t:.4f}")

    # 5. Kalibre Edilmiş Test Logitleri ve Sonrası Ölçümler
    with torch.no_grad():
        test_logitler_kalibre = kalibrator(test_logitler)

    metrikler_kalibre = KalibrasyonMetrikleri.hesapla_tum_metrikler(test_logitler_kalibre, test_etiketler, n_bins=15)
    print(f"\n  [Kalibrasyon Sonrası Test Sonuçları]")
    print(f"  ✓ Kalibre Model Test Doğruluğu: %{metrikler_kalibre['dogruluk']:.2f} (Doğruluk Değişmedi!)")
    print(f"  ✓ Kalibre Model Test ECE: %{metrikler_kalibre['ece']:.2f} (Azalma: %{metrikler_ham['ece'] - metrikler_kalibre['ece']:.2f})")
    print(f"  ✓ Kalibre Model Test NLL: {metrikler_kalibre['nll']:.4f}")
    print(f"  ✓ Kalibre Model Brier Skoru: {metrikler_kalibre['brier_score']:.4f}")

    # 6. Sıcaklık Taraması (NLL Kayıp Yüzeyi)
    t_aralik = np.linspace(0.3, 4.0, 50)
    nll_tarama = []
    for t_val in t_aralik:
        with torch.no_grad():
            olcek = test_logitler / t_val
            nll_tarama.append(F.cross_entropy(olcek, test_etiketler).item())

    sicaklik_tarama = {
        "t_degerleri": t_aralik,
        "nll_degerleri": np.array(nll_tarama)
    }

    # 7. Görselleştirme
    print(f"\n[4/4] 6 Panelli Kalibrasyon Teşhis Panosu Kaydediliyor...")
    gorsellestirici = KalibrasyonGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "calibration_uncertainty_paneli.png")

    gorsellestirici.olustur_kalibrasyon_paneli(
        onceki_metrikler=metrikler_ham,
        sonraki_metrikler=metrikler_kalibre,
        optimal_sicaklik=optimal_t,
        sicaklik_tarama=sicaklik_tarama,
        kayit_yolu=cikti_yolu
    )

    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 84: Probability Calibration & ECE Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
