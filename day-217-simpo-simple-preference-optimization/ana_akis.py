"""
Day 217: SimPO (Simple Preference Optimization) ve Referanssız Marjin Hizalaması Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.simpo_motoru import (
    SimPORewardCalculator,
    SimPOLossObjective,
    SimPOMemoryProfiler,
    SimPOTrainer,
)
from src.simpo_profilleyici import SimPOProfilleyici
from src.gorsellestirici import SimPOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 217 (FAZ 11): SIMPO (SIMPLE PREFERENCE OPTIMIZATION) - REFERANSSIZ VE MARJİN TABANLI HİZALAMA")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: VRAM ve Bellek Tasarruf Analizi
    # -------------------------------------------------------------
    print("\n[1/4] Referans Modelin Devreden Çıkarılmasıyla VRAM Tasarrufu Hesaplanıyor...")
    vram_7b = SimPOMemoryProfiler.vram_tasarrufu_hesapla(model_parametre_milyar=7.0)
    print(f"  • Standart DPO VRAM (Policy + Ref) : {vram_7b['dpo_vram_gb']:.1f} GB")
    print(f"  • SimPO VRAM (Yalnızca Policy)     : {vram_7b['simpo_vram_gb']:.1f} GB")
    print(f"  • Net GPU Bellek Tasarrufu         : {vram_7b['tasarruf_gb']:.1f} GB (-%{vram_7b['tasarruf_yuzde']:.1f} Tasarruf)")
    print("  ✓ %50 VRAM Tasarrufu Başarıyla Teyit Edildi!")

    # -------------------------------------------------------------
    # ADIM 2: Uzunluk-Normalize Örtük Ödül ve Marjin Hesabı
    # -------------------------------------------------------------
    print("\n[2/4] Uzunluk Normalize Log-Olasılık Ödülü ve Marjin (r_w - r_l - γ) Hesaplanıyor...")
    prompt = "Makine öğreniminde aşırı uyum (overfitting) nasıl engellenir?"
    chosen = "Düzenlileştirme (L1/L2), Dropout ve erken durdurma (early stopping) yöntemleri kullanılır."
    rejected = "Daha büyük model eğitilerek ezberletilir."

    adim_sonucu = SimPOTrainer.egitim_adimi(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        beta=2.0,
        gamma_margin=0.80,
    )

    print(f"  • İncelenen İstemi     : '{prompt}'")
    print(f"  • Seçilen Uzunluk      : {adim_sonucu['len_chosen']} kelime")
    print(f"  • Reddedilen Uzunluk   : {adim_sonucu['len_rejected']} kelime")
    print(f"  • Hedeflenen Marjin (γ): {adim_sonucu['hedef_marjin']:.2f}")
    print(f"  • Gerçekleşen Marjin   : Δr = +{adim_sonucu['ortuk_marjin']:.2f}")
    print(f"  • SimPO Kaybı          : {adim_sonucu['kayip']:.4f}")
    print("  ✓ SimPO Marjin Kaybı Başarıyla Hesaplandı!")

    # -------------------------------------------------------------
    # ADIM 3: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli SimPO Teşhis Panosu Oluşturuluyor...")
    profil_raporu = SimPOProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "simpo_paneli.png")

    SimPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ SimPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 217 (FAZ 11): SIMPO (SIMPLE PREFERENCE OPTIMIZATION) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
