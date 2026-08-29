"""
Day 209: Rejection Sampling & Best-of-N Sıcaklık Örneklemesi ve Çoklu Düşünce Filtreleme Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rejection_sampling_motoru import (
    PolicySampler,
    RejectionFilter,
    RSSFTDatasetBuilder,
    SimplePolicyModel,
    RSSFTTrainer,
)
from src.rejection_profilleyici import RejectionProfilleyici
from src.gorsellestirici import RejectionGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 209 (FAZ 11): REJECTION SAMPLING & BEST-OF-N (TEMPERATURE SAMPLING & SFT FILTERING ENGINE)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Sıcaklık Tabanlı Çoklu Aday Üretimi (Policy Sampling)
    # -------------------------------------------------------------
    print("\n[1/4] Sıcaklık (T=0.8) Tabanlı K=8 Aday Düşünce Yolu Örnekleniyor...")
    prompt_ornek = "Bir kutudaki 12 bilyeden 4'ü mavidir. Rastgele seçilen 2 bilyenin mavi olma olasılığı nedir?"
    adaylar = PolicySampler.orneklem_uret(prompt_ornek, k_orneklem=8, sicaklik=0.8)

    print(f"  • Soru Promptu : '{prompt_ornek}'")
    print(f"  • Üretilen Aday: {len(adaylar)} adet paralel düşünce zinciri")
    for a in adaylar[:3]:
        print(f"    - [Aday {a['aday_id']}] Kalite Skoru: {a['odul_skoru']:.2f} | Durum: {'✅ Doğru' if a['dogru_mu'] else '❌ Hatalı'}")
    print("  ✓ Çoklu Düşünce Örneklemesi Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 2: Doğrulayıcı Filtreleme ve Best-of-K Seçimi
    # -------------------------------------------------------------
    print("\n[2/4] Kalite Eşiğine (τ=0.60) Göre Rejection Filtreleme Yürütülüyor...")
    filtre_sonucu = RejectionFilter.adaylari_filtrele_ve_sec(adaylar, esik_skoru=0.60)

    print(f"  • Kabul Edilen Aday Sayısı : {filtre_sonucu['kabul_sayisi']} / {filtre_sonucu['toplam_aday']}")
    print(f"  • Reddedilen Aday Sayısı   : {filtre_sonucu['red_sayisi']}")
    print(f"  • Filtre Kabul Oranı (α)   : %{filtre_sonucu['kabul_orani']*100:.1f}")
    print(f"  • Seçilen En İyi Aday (y*) : Aday {filtre_sonucu['en_iyi_aday']['aday_id']} (Skor: {filtre_sonucu['en_iyi_aday']['odul_skoru']:.2f})")
    print("  ✓ Rejection Filtreleme Başarıyla Gerçekleştirildi!")

    # -------------------------------------------------------------
    # ADIM 3: Sentetik SFT Veri Seti İnşası ve SFT Eğitimi
    # -------------------------------------------------------------
    print("\n[3/4] Sentetik SFT Veri Seti Oluşturuluyor ve Model Eğitiliyor...")
    problem_havuzu = [
        "Problem A: Polinom Çarpanlara Ayırma",
        "Problem B: Geometri Alan Hesabı",
        "Problem C: Kombinatorik Permütasyon",
        "Problem D: Diferansiyel Denklem",
    ]
    sft_verisi = RSSFTDatasetBuilder.sentetik_veri_seti_olustur(problem_havuzu, k_orneklem=8, sicaklik=0.8)
    print(f"  • İşlenen Problem Sayısı   : {sft_verisi['problem_sayisi']}")
    print(f"  • Oluşturulan SFT Çifti    : {sft_verisi['sft_ornek_sayisi']} adet yüksek kaliteli örnek")
    print(f"  • Problem Kapsama Oranı    : %{sft_verisi['problem_kapsama_orani']*100:.1f}")

    # Model SFT Eğitimi
    trainer = RSSFTTrainer()
    dummy_input = torch.randint(0, 128, (4, 16))
    dummy_target = torch.randint(0, 128, (4, 16))
    egitim_metrik = trainer.egitim_adimi(dummy_input, dummy_target)
    print(f"  • SFT Cross-Entropy Kaybı  : {egitim_metrik['sft_loss']:.4f}")
    print(f"  • Model Perplexity (PPL)   : {egitim_metrik['perplexity']:.2f}")
    print("  ✓ RS-SFT Sentetik Eğitimi Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Rejection Sampling Teşhis Panosu Oluşturuluyor...")
    profil_raporu = RejectionProfilleyici.profil_raporu_uret()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "rejection_sampling_paneli.png")

    RejectionGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Rejection Sampling Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 209 (FAZ 11): REJECTION SAMPLING & BEST-OF-N BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
