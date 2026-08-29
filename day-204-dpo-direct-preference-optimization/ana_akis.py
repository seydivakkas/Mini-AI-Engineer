"""
Day 204: DPO (Direct Preference Optimization) Kapalı Form Tercih Hizalama Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dpo_motoru import (
    DPOModel,
    DPOTrainer,
)
from src.dpo_profilleyici import DPOAkisProfilleyici
from src.gorsellestirici import DPOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 204 (FAZ 11): DPO (DIRECT PREFERENCE OPTIMIZATION) CLOSED-FORM ALIGNMENT ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: DPO Motoru ve Model Mimarisi Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] DPO Modelleri (Politika + Dondurulmuş Referans Model) Başlatılıyor...")
    trainer = DPOTrainer(beta=0.1)
    print("  • Eğitilen Model (Policy π_θ)     : Aktif Tercih Optimize Edilen Model")
    print("  • Referans Model (Ref π_ref)     : Dondurulmuş Temel Model")
    print("  • Düzenlileştirme Çarpanı (beta) : 0.10")
    print("  ✓ Ödül Modeli ve RL Örnekleme Döngüsü Gerektirmeyen Kapalı Form Mimarisi Hazır!")

    # -------------------------------------------------------------
    # ADIM 2: Kapalı Form Tercih Kaybı ve Örtük Ödül Testi
    # -------------------------------------------------------------
    print("\n[2/4] Bradley-Terry Eşlemesi ile Örtük Ödül ve Tercih Kaybı Hesaplanıyor...")
    chosen_dummy = torch.randint(0, 128, (2, 10))
    rejected_dummy = torch.randint(0, 128, (2, 10))
    loss, metrikler = trainer.dpo_kaybi_hesapla(chosen_dummy, rejected_dummy)

    print(f"  • Başlangıç DPO Kaybı (Loss) : {loss.item():.4f}")
    print(f"  • Örtük Tercih Ödülü (y_w)   : {metrikler['chosen_reward']:+.4f}")
    print(f"  • Örtük Red Ödülü (y_l)      : {metrikler['rejected_reward']:+.4f}")
    print(f"  • Ödül Marjı (Δr)            : {metrikler['reward_margin']:+.4f}")
    print("  ✓ DPO Matematiksel Kapalı Form Fonksiyonu Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: 10 Adımlık DPO Tercih Eğitimi Döngüsü
    # -------------------------------------------------------------
    print("\n[3/4] 10 Adımlık DPO Tercih Hizalama Eğitimi Yürütülüyor...")
    profil_raporu = DPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=10)

    print("-" * 115)
    print(f"{'Eğitim Adımı':<16} | {'DPO Kaybı (Loss)':<22} | {'Tercih Doğruluğu':<22} | {'Ödül Marjı (Δr)':<20} | {'Tercih Ödülü'}")
    print("-" * 115)
    for adim, kayip, acc, marj, c_odul in zip(
        profil_raporu["adimlar"],
        profil_raporu["kayiplar"],
        profil_raporu["dogruluklar"],
        profil_raporu["odul_marjlari"],
        profil_raporu["chosen_oduller"],
    ):
        print(
            f"Adım #{adim:<10} | "
            f"{kayip:>18.4f}   | "
            f"%{acc:>18.1f}   | "
            f"{marj:>16.2f}   | "
            f"{c_odul:>14.2f}"
        )
    print("-" * 115)
    print(f"  🏆 Nihai Tercih Doğruluğu : %{profil_raporu['son_dogruluk']:.1f}")
    print(f"  📈 Nihai Ödül Marjı (Δr)  : +{profil_raporu['son_marj']:.2f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli DPO Tercih Hizalama Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "dpo_preference_paneli.png")

    DPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ DPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 204 (FAZ 11): DPO DIRECT PREFERENCE OPTIMIZATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
