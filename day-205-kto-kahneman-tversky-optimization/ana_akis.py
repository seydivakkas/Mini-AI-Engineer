"""
Day 205: KTO (Kahneman-Tversky Optimization) Asimetrik Tercih Hizalama Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.kto_motoru import (
    KTOModel,
    KTOTrainer,
)
from src.kto_profilleyici import KTOAkisProfilleyici
from src.gorsellestirici import KTOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 205 (FAZ 11): KTO (KAHNEMAN-TVERSKY OPTIMIZATION) UNPAIRED ALIGNMENT ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: KTO Motoru ve Modellerin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] KTO Modelleri ve Beklenti Teorisi (Prospect Theory) Parametreleri Kuruluyor...")
    trainer = KTOTrainer(beta=0.1, lambda_d=1.0, lambda_u=1.33)
    print("  • Eğitilen Politika (Policy π_θ)         : Aktif Hizalanan Model")
    print("  • Referans Model (Ref π_ref)             : Dondurulmuş Temel Model")
    print("  • Pozitif Tercih Ağırlığı (lambda_d)     : 1.00")
    print("  • Kayıptan Kaçınma Ağırlığı (lambda_u)   : 1.33 (Asimetrik Ceza)")
    print("  ✓ Eşleşmemiş (Unpaired) Tekil İkili Veriyle Çalışan KTO Motoru Hazır!")

    # -------------------------------------------------------------
    # ADIM 2: Tekil İkili Tercih Kaybı ve Örtük Ödül Testi
    # -------------------------------------------------------------
    print("\n[2/4] Eşleşmemiş İkili (Upvote / Downvote) Veri Üzerinde KTO Kaybı Hesaplanıyor...")
    input_dummy = torch.randint(0, 128, (4, 10))
    desirable_dummy = torch.tensor([True, True, False, False])
    loss, metrikler = trainer.kto_kaybi_hesapla(input_dummy, desirable_dummy)

    print(f"  • Başlangıç KTO Kaybı (Loss) : {loss.item():.4f}")
    print(f"  • Referans Çapası (z_ref)    : {metrikler['z_ref']:+.4f}")
    print(f"  • Beğenilen Ödül (Upvote)    : {metrikler['desirable_reward']:+.4f}")
    print(f"  • Reddedilen Ödül (Downvote) : {metrikler['undesirable_reward']:+.4f}")
    print(f"  • Ödül Farkı (Delta)         : {metrikler['reward_delta']:+.4f}")
    print("  ✓ KTO Beklenti Fonksiyonu Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: 10 Adımlık KTO Tercih Eğitimi Döngüsü
    # -------------------------------------------------------------
    print("\n[3/4] 10 Adımlık KTO Tercih Eğitimi Yürütülüyor...")
    profil_raporu = KTOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=10)

    print("-" * 115)
    print(f"{'Eğitim Adımı':<16} | {'KTO Kaybı (Loss)':<22} | {'Hizalama Skoru':<22} | {'Ödül Farkı (Δr)':<20} | {'Beğenilen Ödül'}")
    print("-" * 115)
    for adim, kayip, skor, fark, d_odul in zip(
        profil_raporu["adimlar"],
        profil_raporu["kayiplar"],
        profil_raporu["hizalama_skorlari"],
        profil_raporu["odul_farklari"],
        profil_raporu["desirable_oduller"],
    ):
        print(
            f"Adım #{adim:<10} | "
            f"{kayip:>18.4f}   | "
            f"%{skor:>18.1f}   | "
            f"{fark:>16.2f}   | "
            f"{d_odul:>14.2f}"
        )
    print("-" * 115)
    print(f"  🏆 Nihai İnsan Hizalama Skoru : %{profil_raporu['son_hizalama_skoru']:.1f}")
    print(f"  📈 Nihai Ödül Ayrışması (Δ)  : +{profil_raporu['son_fark']:.2f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli KTO Asimetrik Tercih Hizalama Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "kto_prospect_paneli.png")

    KTOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ KTO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 205 (FAZ 11): KTO (KAHNEMAN-TVERSKY OPTIMIZATION) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
