"""
Day 85: Enerji Tabanlı OOD Tespiti ve Seçici Tahmin Laboratuvarı
---------------------------------------------------------------
Serbest enerji (Free Energy) skoru ile Dağılım Dışı (OOD) tespiti,
Softmax MSP karşılaştırması, AUROC/FPR95 metrikleri ve Seçici Çekimserlik (Abstention) akışı.

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

from src.model import VisionOODModeli
from src.enerji_ood import EnerjiTabanliOODDedektoru
from src.secmeli_tahminci import SecmeliTahminci
from src.metrikler import OODMetrikleri
from src.gorsellestirici import OODGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def id_verisi_olustur(ornek_sayisi: int = 600, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    """Dağılım İçi (In-Distribution) veri kümesi"""
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 0.8
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        x[i, :, (c%4)*7:(c%4)*7+6, (c//4)*7:(c//4)*7+6] += 1.8
    return x, y


def ood_verisi_olustur(ornek_sayisi: int = 200, gorsel_boyutu: int = 32):
    """Dağılım Dışı (Out-of-Distribution) gürültü ve farklı frekans örüntüleri"""
    x_ood = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 2.5
    # Çizgisel ve frekans desenleri ekle (ID sınıflarında bulunmayan desenler)
    for i in range(ornek_sayisi):
        freq = (i % 5) + 1
        grid = torch.sin(torch.linspace(0, freq * np.pi, gorsel_boyutu)).unsqueeze(0).repeat(gorsel_boyutu, 1)
        x_ood[i, 0] += grid * 2.0
    y_ood = torch.full((ornek_sayisi,), fill_value=-1)
    return x_ood, y_ood


def main():
    print("=" * 85)
    print("🚀 Day 85: Enerji Tabanlı OOD Tespiti ve Seçici Tahmin (Abstention) Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Kümelerinin Oluşturulması
    tr_x, tr_y = id_verisi_olustur(640, 10)
    val_x, val_y = id_verisi_olustur(160, 10)
    test_id_x, test_id_y = id_verisi_olustur(200, 10)
    test_ood_x, test_ood_y = ood_verisi_olustur(200)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)
    test_id_loader = DataLoader(TensorDataset(test_id_x, test_id_y), batch_size=32, shuffle=False)
    test_ood_loader = DataLoader(TensorDataset(test_ood_x, test_ood_y), batch_size=32, shuffle=False)

    # 2. Modelin Eğitimi
    print(f"\n[1/4] Vision Modeli ID Verisi Üzerinde Eğitiliyor...")
    model = VisionOODModeli(sinif_sayisi=10, taban_kanal=32).to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for _ in range(8):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(cihaz), by.to(cihaz)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()

    # 3. Logitlerin Toplanması
    print(f"\n[2/4] ID ve OOD Logitleri Çıkarılıyor...")
    val_id_logits, _ = VisionOODModeli.logit_cikar(model, val_loader, cihaz)
    test_id_logits, test_id_labels = VisionOODModeli.logit_cikar(model, test_id_loader, cihaz)
    test_ood_logits, _ = VisionOODModeli.logit_cikar(model, test_ood_loader, cihaz)

    # 4. Enerji Skoru vs Softmax MSP Karşılaştırması
    print(f"\n[3/4] Enerji Skoru vs Softmax MSP ile OOD Tespiti Kıyaslanıyor...")
    id_enerji = EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(test_id_logits, sicaklik=1.0).numpy()
    ood_enerji = EnerjiTabanliOODDedektoru.enerji_skoru_hesapla(test_ood_logits, sicaklik=1.0).numpy()

    id_msp = EnerjiTabanliOODDedektoru.msp_skoru_hesapla(test_id_logits).numpy()
    ood_msp = EnerjiTabanliOODDedektoru.msp_skoru_hesapla(test_ood_logits).numpy()

    metrikler_enerji = OODMetrikleri.hesapla_ood_metrikleri(id_enerji, ood_enerji)
    metrikler_msp = OODMetrikleri.hesapla_ood_metrikleri(id_msp, ood_msp)

    print(f"  ✓ Enerji Skoru  ──> AUROC: %{metrikler_enerji['auroc']:.2f} | FPR95: %{metrikler_enerji['fpr95']:.2f} | AUPR: %{metrikler_enerji['aupr']:.2f}")
    print(f"  ✓ Softmax (MSP) ──> AUROC: %{metrikler_msp['auroc']:.2f} | FPR95: %{metrikler_msp['fpr95']:.2f} | AUPR: %{metrikler_msp['aupr']:.2f}")

    # 5. Seçici Tahmin (Abstention) Değerlendirmesi
    print(f"\n[4/4] Seçici Tahmin (Abstention) Mekanizması Çalıştırılıyor...")
    dedektor = EnerjiTabanliOODDedektoru(sicaklik=1.0)
    esik_gamma = dedektor.esik_belirle(val_id_logits, hedef_tpr=0.95)
    print(f"  ✓ Val Kümesinden Belirlenen Güvenlik Eşiği (γ): {esik_gamma:.4f}")

    secmeli = SecmeliTahminci(esik_degeri=esik_gamma, skor_tipi="enerji", sicaklik=1.0)
    secmeli_rapor = secmeli.secmeli_tahmin_yap(test_id_logits)

    ham_tahminler = test_id_logits.argmax(dim=-1)
    ham_dogru = (ham_tahminler == test_id_labels).sum().item()
    ham_hata_orani = (1.0 - (ham_dogru / len(test_id_labels))) * 100.0

    kabul_mask = secmeli_rapor["kabul_maskesi"]
    kabul_dogru = (ham_tahminler[kabul_mask] == test_id_labels[kabul_mask]).sum().item()
    kabul_toplam = kabul_mask.sum().item()
    filtreli_hata_orani = (1.0 - (kabul_dogru / max(1, kabul_toplam))) * 100.0

    print(f"  ✓ Kapsam (Coverage): %{secmeli_rapor['kapsam_orani']:.2f} ({kabul_toplam}/{len(test_id_labels)} örnek kabul edildi)")
    print(f"  ✓ Filtresiz Ham Hata Oranı: %{ham_hata_orani:.2f} ──> Seçici Tahmin Hata Oranı: %{filtreli_hata_orani:.2f}")

    kapsam_risk = SecmeliTahminci.kapsam_risk_egrisi(test_id_logits, test_id_labels, "enerji", 1.0)

    # 6. Teşhis Panosu
    gorsellestirici = OODGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ood_selective_prediction_paneli.png")

    gorsellestirici.olustur_ood_paneli(
        id_enerji=id_enerji,
        ood_enerji=ood_enerji,
        metrikler_enerji=metrikler_enerji,
        metrikler_msp=metrikler_msp,
        kapsam_risk_enerji=kapsam_risk,
        esik_degeri=esik_gamma,
        ham_hata_orani=ham_hata_orani,
        filtreli_hata_orani=filtreli_hata_orani,
        kayit_yolu=cikti_yolu
    )

    print(f"\n  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 85: Enerji Tabanlı OOD ve Seçici Tahmin Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
