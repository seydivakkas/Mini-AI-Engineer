"""
Day 211: Çok Turlu (Multi-Turn) Diyalog RLHF ve Zamansal Kredi Dağılımı Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dialogue_rlhf_motoru import (
    DialogueState,
    UserSimulator,
    MultiTurnRewardModel,
    TemporalCreditAssigner,
    MultiTurnRLHFTrainer,
)
from src.dialogue_profilleyici import DialogueProfilleyici
from src.gorsellestirici import DialogueGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 211 (FAZ 11): MULTI-TURN DIALOGUE RLHF & TEMPORAL CREDIT ASSIGNMENT ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Çok Turlu Diyalog Simülasyonu
    # -------------------------------------------------------------
    print("\n[1/4] Çok Turlu Diyalog ve Kullanıcı Simülasyonu Başlatılıyor...")
    trainer = MultiTurnRLHFTrainer(gamma=0.95)
    sonuc = trainer.tam_diyalog_yurut()

    print(f"  • Toplam Konuşma Turu : {sonuc['toplam_tur']}")
    print(f"  • Kullanıcı Hedefi    : Veritabanı İndeks Optimizasyonu")
    print(f"  • Hedef Tamamlandı mı : {'✅ BAŞARILI' if sonuc['hedef_basarildi_mi'] else '❌ BAŞARISIZ'}")
    print("  ✓ Diyalog Akışı Başarıyla Yürütüldü!")

    # -------------------------------------------------------------
    # ADIM 2: Adım Bazlı Ödüller ve Konuşma Dökümü
    # -------------------------------------------------------------
    print("\n[2/4] Çok Turlu Konuşma Dökümü ve Anlık Ödüller (r_t)...")
    for adim in sonuc["diyalog_adimlari"]:
        print(f"  • [Tur {adim['tur']}]")
        print(f"    - Kullanıcı : {adim['user']}")
        print(f"    - Asistan   : {adim['assistant']}")
        print(f"    - Tur Ödülü : r_{adim['tur']} = {adim['ara_odul']:+.2f}")

    print(f"  • Terminal Hedef Ödülü : R_T = +{sonuc['terminal_odul']:.2f}")

    # -------------------------------------------------------------
    # ADIM 3: Zamansal Kredi Dağılımı (Temporal Credit Assignment)
    # -------------------------------------------------------------
    print("\n[3/4] Zamansal İndirimli Birikimli Getiriler (G_t = r_t + γ*G_{t+1}) Hesaplanıyor...")
    print("-" * 80)
    print(f"{'Tur No':<10} | {'Anlık Ödül (r_t)':<22} | {'İndirimli Getiri (G_t)'}")
    print("-" * 80)
    for i, (r, g) in enumerate(zip(sonuc["tur_odulleri"], sonuc["indirimli_getiriler"]), 1):
        print(f"Tur {i:<6} | {r:>+18.2f}    | {g:>+20.4f}")
    print("-" * 80)
    print(f"  🏆 İlk Tur Getirisi (G_1) : {sonuc['indirimli_getiriler'][0]:+.4f} (İlk stratejik adımlara kredi aktarıldı)")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Çok Turlu Diyalog RLHF Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DialogueProfilleyici.profil_raporu_uret()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "multi_turn_rlhf_paneli.png")

    DialogueGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Diyalog RLHF Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 211 (FAZ 11): ÇOK TURLU DİYALOG RLHF BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
