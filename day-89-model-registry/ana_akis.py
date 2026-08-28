"""
Day 89: Model Kayıt Sistemi, Model Sürümleme ve Staging/Production Yaşam Döngüsü Laboratuvarı
---------------------------------------------------------------------------------------------
Model Sürümleme (v1 -> v2 -> v3), Otomatik Kalite Kapısı (Quality Gate),
Aşama Geçişleri (Staging -> Production -> Archived) ve Sıfır Kesintili Geri Alma (Rollback) Akışı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.kayit_motoru import ModelKayitMotoru
from src.model import UretimVisionModeli
from src.kalite_kapisi import ModelKaliteKapisi
from src.gorsellestirici import RegistryGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def veri_olustur(ornek_sayisi: int = 600, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 0.5
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        r = (c % 3) * 9 + 2
        col = (c // 3) * 9 + 2
        x[i, :, r:r+8, col:col+8] += 2.0
    return x, y


def main():
    print("=" * 85)
    print("🚀 Day 89: Model Kayıt Sistemi, Sürümleme ve Staging/Production Yaşam Döngüsü Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri ve Motor Hazırlığı
    tr_x, tr_y = veri_olustur(600, 10)
    val_x, val_y = veri_olustur(200, 10)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)

    depo_yolu = os.path.join(os.path.dirname(__file__), ".model_registry")
    motor = ModelKayitMotoru(depo_dizini=depo_yolu)
    kalite_kapisi = ModelKaliteKapisi(min_dogruluk=90.0, max_gecikme_ms=30.0, max_ece=0.15)
    model_adi = "MiniViT-Production-Classifier"

    # -------------------------------------------------------------
    # ADIM 1: Model v1 (İlk Temel Sürüm) Eğitimi ve Kaydı
    # -------------------------------------------------------------
    print(f"\n[1/4] Model v1 (İlk Temel Model) Eğitiliyor ve Registry'ye Ekleniyor...")
    model_v1 = UretimVisionModeli(sinif_sayisi=10, taban_kanal=32)
    UretimVisionModeli.egit(model_v1, tr_loader, epok_sayisi=6, lr=2e-3, cihaz=cihaz)

    v1_agirlik_yolu = os.path.join(depo_yolu, "gecici_v1.pt")
    torch.save(model_v1.state_dict(), v1_agirlik_yolu)

    v1_kalite = kalite_kapisi.degerlendir(model_v1, val_loader, cihaz=cihaz)
    v1_no = motor.surum_ekle(
        model_adi=model_adi,
        kaynak_agirlik_yolu=v1_agirlik_yolu,
        run_id="run_baseline_001",
        metrikler=v1_kalite["metrikler"],
        sema=UretimVisionModeli.model_semasi(),
        etiketler={"developer": "seydivakkas", "architecture": "VisionCNN-32"}
    )
    print(f"  ✓ Model v{v1_no} Kaydedildi! Doğruluk: %{v1_kalite['metrikler']['val_acc']:.2f}")

    # v1'i Staging -> Production terfi ettir
    motor.asama_degistir(model_adi, v1_no, "STAGING", aciklama="v1 Staging testlerine alındı")
    motor.asama_degistir(model_adi, v1_no, "PRODUCTION", aciklama="v1 İlk Üretim Sürümü Olarak Canlıya Alındı")
    print(f"  🚀 Model v{v1_no} Başarıyla PRODUCTION Aşamasına Terfi Etti!")

    # -------------------------------------------------------------
    # ADIM 2: Model v2 (Gelişmiş Aday) Eğitimi ve Kalite Kapısı
    # -------------------------------------------------------------
    print(f"\n[2/4] Model v2 (Gelişmiş Aday Model) Eğitiliyor ve Kalite Kapısına Alınıyor...")
    model_v2 = UretimVisionModeli(sinif_sayisi=10, taban_kanal=48)
    UretimVisionModeli.egit(model_v2, tr_loader, epok_sayisi=8, lr=1e-3, cihaz=cihaz)

    v2_agirlik_yolu = os.path.join(depo_yolu, "gecici_v2.pt")
    torch.save(model_v2.state_dict(), v2_agirlik_yolu)

    v2_kalite = kalite_kapisi.degerlendir(model_v2, val_loader, cihaz=cihaz)
    v2_no = motor.surum_ekle(
        model_adi=model_adi,
        kaynak_agirlik_yolu=v2_agirlik_yolu,
        run_id="run_enhanced_002",
        metrikler=v2_kalite["metrikler"],
        sema=UretimVisionModeli.model_semasi(),
        etiketler={"developer": "seydivakkas", "architecture": "VisionCNN-48"}
    )

    print(f"  ✓ Model v{v2_no} Kalite Kapısı Değerlendirmesi: {'GEÇTİ ✅' if v2_kalite['gecti_mi'] else 'KALDI ❌'}")
    print(f"    • Doğruluk: %{v2_kalite['metrikler']['val_acc']:.2f} (Eşik: >= %{v2_kalite['esikler']['min_dogruluk']})")
    print(f"    • Gecikme : {v2_kalite['metrikler']['latency_ms']:.2f} ms (Eşik: <= {v2_kalite['esikler']['max_gecikme_ms']} ms)")
    print(f"    • ECE     : {v2_kalite['metrikler']['ece']:.4f} (Eşik: <= {v2_kalite['esikler']['max_ece']})")

    if v2_kalite["gecti_mi"]:
        motor.asama_degistir(model_adi, v2_no, "PRODUCTION", mevcut_uretimi_arsivle=True, aciklama="v2 Kalite testlerini geçti, v1 arşivlendi")
        print(f"  🚀 Model v{v2_no} PRODUCTION Oldu! (Önceki v{v1_no} Otomatik ARCHIVED Yapıldı).")

    # -------------------------------------------------------------
    # ADIM 3: Model v3 (Kusurlu Aday) ve Kalite Kapısı Reddi
    # -------------------------------------------------------------
    print(f"\n[3/4] Model v3 (Kusurlu Aday Model) Eğitiliyor ve Kalite Kapısı Test Ediliyor...")
    model_v3 = UretimVisionModeli(sinif_sayisi=10, taban_kanal=16)
    # Hatalı/Yetersiz eğitim
    UretimVisionModeli.egit(model_v3, tr_loader, epok_sayisi=1, lr=5e-1, cihaz=cihaz)

    v3_agirlik_yolu = os.path.join(depo_yolu, "gecici_v3.pt")
    torch.save(model_v3.state_dict(), v3_agirlik_yolu)

    v3_kalite = kalite_kapisi.degerlendir(model_v3, val_loader, cihaz=cihaz)
    v3_no = motor.surum_ekle(
        model_adi=model_adi,
        kaynak_agirlik_yolu=v3_agirlik_yolu,
        run_id="run_flawed_003",
        metrikler=v3_kalite["metrikler"],
        sema=UretimVisionModeli.model_semasi(),
        etiketler={"developer": "seydivakkas", "architecture": "VisionCNN-16-Flawed"}
    )
    print(f"  ✓ Model v{v3_no} Kalite Kapısı Değerlendirmesi: {'GEÇTİ ✅' if v3_kalite['gecti_mi'] else 'REDDEDİLDİ ❌'}")
    print(f"    • Doğruluk: %{v3_kalite['metrikler']['val_acc']:.2f} (Eşik Karşılanamadı! Üretime geçiş BLOKE EDİLDİ).")

    # -------------------------------------------------------------
    # ADIM 4: Sıfır Kesintili Acil Geri Alma (Rollback) Simülasyonu
    # -------------------------------------------------------------
    print(f"\n[4/4] Sıfır Kesintili Acil Geri Alma (Instant Rollback) Simülasyonu...")
    print(f"  ⚠️ Canlı ortamda v2 için alarm tetiklendi! Geri alma (Rollback) başlatılıyor...")
    aktif_prod = motor.geri_al(model_adi, aciklama="v2 alarmı sonrası v1'e acil rollback")
    print(f"  ✓ Geri Alma Tamamlandı! Güncel Aktif Üretim Modeli: v{aktif_prod['surum_no']} ({aktif_prod['asama']})")

    # Tüm sürümleri listele
    surumler = motor.tum_surumleri_listele(model_adi)

    # Teşhis Panosunu Oluştur
    gorsellestirici = RegistryGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "model_registry_paneli.png")

    gorsellestirici.olustur_registry_paneli(
        surumler=surumler,
        kalite_raporu_v2=v2_kalite,
        kalite_raporu_v3=v3_kalite,
        kayit_yolu=cikti_yolu
    )

    print(f"\n  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 89: Model Registry & Yaşam Döngüsü Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
