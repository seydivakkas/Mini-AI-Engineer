"""
Day 215: İteratif ve Çevrimiçi DPO (Online Preference Loop) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.iteratif_dpo_motoru import (
    OnlinePreferenceBuffer,
    OnlineRolloutSampler,
    ReferencePolicyUpdater,
    IterativeDPOTrainer,
)
from src.iteratif_dpo_profilleyici import IterativeDPOProfilleyici
from src.gorsellestirici import IterativeDPOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 215 (FAZ 11): İTERATİF VE ÇEVRİMİÇİ DPO (ONLINE PREFERENCE OPTIMIZATION & REFERENCE SWAPPING)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Dinamik Tercih Havuzu Başlatma
    # -------------------------------------------------------------
    print("\n[1/4] Kayan Pencereli Dinamik Tercih Havuzu (Online Buffer) Başlatılıyor...")
    buffer = OnlinePreferenceBuffer(kapasite=500)
    print(f"  • Başlangıç Tercih Havuzu Kapasitesi: {buffer.kapasite} örnek")
    print("  ✓ Tercih Havuzu Başarıyla Hazırlandı!")

    # -------------------------------------------------------------
    # ADIM 2: 3-Turlu İteratif Örnekleme ve Referans Güncelleme
    # -------------------------------------------------------------
    print("\n[2/4] 3-Turlu İteratif DPO Döngüsü Yürütülüyor...")
    prompt = "Bir kuantum bilgisayarın klasik bilgisayara üstünlüğünü kısaca açıklayın."

    for tur in range(1, 4):
        sonuc = IterativeDPOTrainer.iteratif_tur_yurut(prompt, tur, buffer)
        print(f"\n  --- [TUR {tur}] İterasyon Adımı ---")
        print(f"  • Seçilen (Chosen)   : '{sonuc['chosen']}'")
        print(f"  • Reddedilen (Reject): '{sonuc['rejected']}'")
        print(f"  • Online DPO Kaybı   : {sonuc['kayip']:.4f}")
        print(f"  • Örtük Ödül Marjini : Δr = +{sonuc['ortuk_odul_marjini']:.2f}")
        print(f"  • Tercih Havuzu Yükü : {sonuc['buffer_boyutu']} çift")

    print("\n  ✓ 3 Turlu İteratif DPO Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 3: Referans Modeli Kaydırma (Ref Swapping) Doğrulaması
    # -------------------------------------------------------------
    print("\n[3/4] Referans Model Kaydırma (Reference Swapping) Denetleniyor...")
    mevcut_agirliklar = {"katman_1": 0.45, "katman_2": 0.88}
    yeni_referans = ReferencePolicyUpdater.referansi_guncelle(mevcut_agirliklar)
    print(f"  • Yeni Referans Model: π_ref <- π_θ_3 ({yeni_referans})")
    print("  ✓ Dağılım Dışı Sapma (OOD Drift) Başarıyla Sıfırlandı!")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli İteratif DPO Teşhis Panosu Oluşturuluyor...")
    profil_raporu = IterativeDPOProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "iteratif_dpo_paneli.png")

    IterativeDPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ İteratif DPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 215 (FAZ 11): İTERATİF VE ÇEVRİMİÇİ DPO BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
