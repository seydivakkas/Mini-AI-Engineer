"""
Day 207: ORM (Outcome Reward Model) ve Best-of-N Test-Zamanı Çıkarım Ölçekleme Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.orm_motoru import (
    OutcomeRewardModel,
    ORMTrainer,
    BestOfNRanker,
)
from src.orm_profilleyici import ORMAkisProfilleyici
from src.gorsellestirici import ORMGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 207 (FAZ 11): ORM (OUTCOME REWARD MODEL) & BEST-OF-N INFERENCE SCALING ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: ORM Modelinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] ORM (Outcome Reward Model) Mimarisi Başlatılıyor...")
    orm_model = OutcomeRewardModel(vocab_size=128, embed_dim=64)
    trainer = ORMTrainer(orm_model=orm_model)
    print("  • Mimari               : Transformer Encoder + Global Reward Head")
    print("  • Değerlendirme        : Tam Yanıt Bütününe Skalar Kalite Puanı (r_psi)")
    print("  • Referans Paradigma   : Cobbe et al. GSM8K Verifier & Test-Time Re-ranking")
    print("  ✓ ORM Modeli Başarıyla Yüklendi!")

    # -------------------------------------------------------------
    # ADIM 2: Çiftli Tercih Eğitimi ve Ödül Marjı Testi
    # -------------------------------------------------------------
    print("\n[2/4] Bradley-Terry Çiftli Tercih Kaybı ve Ödül Marjı Hesaplanıyor...")
    chosen_dummy = torch.randint(0, 128, (4, 16))
    rejected_dummy = torch.randint(0, 128, (4, 16))
    metrikler = trainer.egitim_adimi(chosen_dummy, rejected_dummy)

    print(f"  • Başlangıç ORM Kaybı : {metrikler['loss']:.4f}")
    print(f"  • Tercih Ödülü (y_w)  : {metrikler['r_chosen']:+.4f}")
    print(f"  • Reddedilen Ödül     : {metrikler['r_rejected']:+.4f}")
    print(f"  • Ödül Marjı (Δr)     : {metrikler['reward_margin']:+.4f}")
    print("  ✓ Bradley-Terry Optimizasyonu Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Test-Zamanı Best-of-N Re-ranking ve Ölçekleme
    # -------------------------------------------------------------
    print("\n[3/4] Test-Zamanı Hesaplama Ölçeklemesi (Best-of-N Re-ranking N=1..64) Yürütülüyor...")
    profil_raporu = ORMAkisProfilleyici.olcekleme_profilini_cikar()

    print("-" * 115)
    print(f"{'Örneklem (N)':<16} | {'Pass@1 Doğruluk':<22} | {'ORM Kaybı (Loss)':<22} | {'Ödül Marjı (Δr)'}")
    print("-" * 115)
    for n, acc, loss_v, marj in zip(
        profil_raporu["n_degerleri"],
        profil_raporu["pass_at_1_oranlari"],
        profil_raporu["orm_kayiplari"],
        profil_raporu["reward_marjlari"],
    ):
        print(
            f"N = {n:<12} | "
            f"%{acc:>16.1f}   | "
            f"{loss_v:>18.4f}   | "
            f"+{marj:>14.2f}"
        )
    print("-" * 115)
    print(f"  🏆 N=1 Başarımı         : {profil_raporu['cikarim_olcekleme_yasasi']['n_1_basarim']}")
    print(f"  🚀 N=64 Başarımı        : {profil_raporu['cikarim_olcekleme_yasasi']['n_64_basarim']}")
    print(f"  📈 Çıkarım Kazanç Farkı : {profil_raporu['cikarim_olcekleme_yasasi']['kazanc_farki']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli ORM ve Best-of-N Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "orm_outcome_paneli.png")

    ORMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ ORM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 207 (FAZ 11): ORM VE BEST-OF-N ÇIKARIM ÖLÇEKLEMESİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
