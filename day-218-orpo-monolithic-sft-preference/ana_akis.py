"""
Day 218: ORPO (Odds Ratio Preference Optimization) ve Monolitik Tercih Hizalaması Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.orpo_motoru import (
    SequenceOddsCalculator,
    ORPOLossObjective,
    MonolithicPipelineProfiler,
    ORPOTrainer,
)
from src.orpo_profilleyici import ORPOProfilleyici
from src.gorsellestirici import ORPOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 218 (FAZ 11): ORPO (ODDS RATIO PREFERENCE OPTIMIZATION) - MONOLİTİK SFT VE TERCİH HİZALAMASI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Monolitik Süreç Tasarruf Analizi
    # -------------------------------------------------------------
    print("\n[1/4] İki Aşamalı (SFT+DPO) vs Tek Aşamalı Monolitik ORPO Süreç Kıyaslaması...")
    sure_raporu = MonolithicPipelineProfiler.egitim_sureleri_kiyasla(50000)
    print(f"  • İki Aşamalı Süre (SFT -> DPO) : {sure_raporu['sft_dpo_iki_asama_saat']:.1f} GPU Saati")
    print(f"  • Monolitik ORPO Tek Aşama Süresi: {sure_raporu['orpo_tek_asama_saat']:.1f} GPU Saati")
    print(f"  • Net Zaman ve GPU Tasarrufu   : {sure_raporu['tasarruf_saat']:.1f} Saat (-%{sure_raporu['tasarruf_yuzde']:.1f} Hızlanma)")
    print("  ✓ Tek Aşamalı Mimari Başarıyla Teyit Edildi!")

    # -------------------------------------------------------------
    # ADIM 2: Odds Ratio ve Monolitik Kayıp (L_SFT + λ*L_OR)
    # -------------------------------------------------------------
    print("\n[2/4] Dizilim Oranı (Odds Ratio) ve Monolitik Kayıp Hesaplanıyor...")
    prompt = "Yapay zekada transfer öğrenimi (transfer learning) nedir?"
    chosen = "Önceden büyük veri setlerinde eğitilmiş bir modelin ağırlıklarını yeni bir alt görev için uyarlamaktır."
    rejected = "Bilgisayarların birbirine USB ile veri kopyalamasıdır."

    adim_sonucu = ORPOTrainer.egitim_adimi(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        lambda_or=0.10,
    )

    print(f"  • İncelenen İstemi : '{prompt}'")
    print(f"  • SFT NLL Kaybı     : {adim_sonucu['l_sft']:.4f} (Talimat Öğrenimi)")
    print(f"  • Odds Ratio Kaybı  : {adim_sonucu['l_or']:.4f} (Tercih Ayrıştırması)")
    print(f"  • Odds Ratio Oranı  : OR = {adim_sonucu['odds_ratio']:.2f} (Chosen / Rejected)")
    print(f"  • Toplam ORPO Kaybı : {adim_sonucu['l_orpo_toplam']:.4f}")
    print("  ✓ Monolitik ORPO Kayıp Adımı Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 3: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli ORPO Teşhis Panosu Oluşturuluyor...")
    profil_raporu = ORPOProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "orpo_paneli.png")

    ORPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ ORPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 218 (FAZ 11): ORPO (ODDS RATIO PREFERENCE OPTIMIZATION) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
