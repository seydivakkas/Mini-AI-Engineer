"""
Day 92: Eğitim Öncesi Veri Sözleşmesi ve Hazır Bulunuşluk Laboratuvarı
---------------------------------------------------------------------
Şema, Boyut, NaN/Inf, Sınıf Dengesizliği ve Train-Val Sızıntısı Denetimi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sozlesme_kurallari import VeriSozlesmesi
from src.sizinti_dedektoru import VeriSizintiDedektoru
from src.veri_denetleyici import VeriDenetleyici
from src.hazir_bulunusluk_kapisi import HazirBulunuslukKapisi
from src.gorsellestirici import VeriSozlesmesiGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 85)
    print("🚀 Day 92: Eğitim Öncesi Veri Sözleşmesi ve Hazır Bulunuşluk (Readiness Gate) Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)

    # 1. Katı Veri Sözleşmesi Tanımı
    sozlesme = VeriSozlesmesi(
        beklenen_kanal=3,
        beklenen_yukseklik=32,
        beklenen_genislik=32,
        beklenen_dtype="float32",
        min_deger_limiti=-4.0,
        maks_deger_limiti=4.0,
        min_ornek_sayisi=100,
        beklenen_sinif_sayisi=10,
        maks_sinif_dengesizlik_orani=8.0,
        nadir_sinif_min_ornek=10,
        maks_sizinti_toleransi=0.0,
    )

    denetleyici = VeriDenetleyici(sozlesme=sozlesme)
    sizinti_dedektoru = VeriSizintiDedektoru()
    kapi = HazirBulunuslukKapisi(sızıntıda_bloke_et=True)

    # -------------------------------------------------------------
    # ADIM 1: Kusursuz (Temiz) Veri Seti Denetimi
    # -------------------------------------------------------------
    print("\n[1/3] SENARYO 1: Kurallara Tam Uyan Temiz Veri Seti Denetleniyor...")
    train_temiz = torch.randn(400, 3, 32, 32).clamp(-3.0, 3.0)
    val_temiz = torch.randn(100, 3, 32, 32).clamp(-3.0, 3.0)
    train_etiket_temiz = torch.randint(0, 10, (400,))

    denetim_temiz = denetleyici.denetle(train_temiz, train_etiket_temiz)
    sizinti_temiz = sizinti_dedektoru.sizinti_tara(train_temiz, val_temiz)
    karar_temiz = kapi.degerlendir(denetim_temiz, sizinti_temiz)

    print(f"  ✓ Toplam Örnek: {denetim_temiz.toplam_ornek} | NaN/Inf: {denetim_temiz.nan_inf_sayisi}")
    print(f"  ✓ Sızıntı Sayısı: {sizinti_temiz.kesisen_ornek_sayisi} (%{sizinti_temiz.sizinti_orani_val * 100:.1f})")
    print(f"  ✓ Kapı Kararı: {karar_temiz.durum.value}")
    print(f"  ✓ Eğitim Başlatılabilir mi: {karar_temiz.egitim_baslatilabilir_mi}")

    # -------------------------------------------------------------
    # ADIM 2: Bozulmuş ve Sızıntılı (Contaminated) Veri Seti
    # -------------------------------------------------------------
    print("\n[2/3] SENARYO 2: Hatalı, NaN İçeren ve Sızıntılı Veri Seti Denetleniyor...")
    train_bozuk = torch.randn(400, 3, 32, 32)
    # NaN ve Sınır Aşımı Enjeksiyonu
    train_bozuk[10, 0, 5, 5] = float("nan")
    train_bozuk[25, 1, 12, 12] = float("inf")
    train_bozuk[50:60] += 8.0  # Sınır aşımı (+8.0)

    # Aşırı sınıf dengesizliği
    etiket_bozuk = torch.tensor([0] * 350 + [1] * 30 + [2] * 20)

    # Train-Val Sızıntısı (Train'den 25 örneği Val setine sızdır)
    val_bozuk = torch.cat([val_temiz[:75], train_bozuk[:25]], dim=0)

    denetim_bozuk = denetleyici.denetle(train_bozuk, etiket_bozuk)
    sizinti_bozuk = sizinti_dedektoru.sizinti_tara(train_bozuk, val_bozuk)
    karar_bozuk = kapi.degerlendir(denetim_bozuk, sizinti_bozuk)

    print(f"  ! Toplam İhlal Sayısı: {karar_bozuk.toplam_ihlal_sayisi}")
    print(f"  ! Sızıntı Sayısı: {sizinti_bozuk.kesisen_ornek_sayisi} (%{sizinti_bozuk.sizinti_orani_val * 100:.1f})")
    print(f"  ! Kapı Kararı: {karar_bozuk.durum.value}")
    print(f"  ! Eğitim Başlatılabilir mi: {karar_bozuk.egitim_baslatilabilir_mi}")

    print("\n🚨 BLOKE EDEN KRİTİK HATALAR:")
    for err in karar_bozuk.bloke_eden_hatalar:
        print(f"  [BLOKE] {err}")

    # -------------------------------------------------------------
    # ADIM 3: 6-Panelli Teşhis Panosunun Üretilmesi
    # -------------------------------------------------------------
    print("\n[3/3] 6-Panelli Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = VeriSozlesmesiGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "veri_sozlesmesi_paneli.png")

    gorsellestirici.olustur_sozlesme_paneli(
        denetim_sonucu=denetim_bozuk,
        sizinti_raporu=sizinti_bozuk,
        kapi_karari=karar_bozuk,
        ornek_tensörler=train_bozuk.numpy(),
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ 6-Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 92: Eğitim Öncesi Veri Sözleşmesi Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
