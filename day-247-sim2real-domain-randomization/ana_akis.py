"""
Day 247: Sim2Real Transferi ve Domain Randomization Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.domain_randomization_motoru import (
    VisualRandomizer,
    DynamicsRandomizer,
    ActionDelayInjector,
    Sim2RealEvaluator,
)
from src.sim2real_profilleyici import Sim2RealProfilleyici
from src.gorsellestirici import Sim2RealGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 247 (FAZ 13): SIM2REAL TRANSFERİ — DOMAIN RANDOMIZATION İLE SIFIR HATA GERÇEK DÜNYA AKTARIMI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Görsel Domain Randomization (Visual DR)
    # -------------------------------------------------------------
    print("\n[1/4] Görsel Domain Randomization (Işık, Doku, Gürültü) Uygulanıyor...")
    orijinal_goruntu = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    orijinal_goruntu[20:44, 20:44] = [0.8, 0.1, 0.1]
    dr_goruntu = VisualRandomizer.randomize_image(orijinal_goruntu, tohum=42)

    print(f"  • Orijinal Görüntü Piksel Ortalaması : {float(np.mean(orijinal_goruntu)):.3f}")
    print(f"  • Randomize Görüntü Piksel Ortalaması: {float(np.mean(dr_goruntu)):.3f}")
    print(f"  • Eklenen Gauss Gürültü Varyansı    : {float(np.var(dr_goruntu - orijinal_goruntu)):.5f}")

    # -------------------------------------------------------------
    # ADIM 2: Dinamik ve Fiziksel Parametre Örneklemesi (Dynamics DR)
    # -------------------------------------------------------------
    print("\n[2/4] Fiziksel ve Dinamik Parametreler Rastgeleleştiriliyor...")
    dinamik = DynamicsRandomizer.sample_dynamics_parameters(tohum=100)
    print(f"  • Sürtünme Katsayısı (μ)          : {dinamik['surtunme_katsayisi']} (Nominal: 0.50)")
    print(f"  • Bağlantı Kütle Çarpanı (Mass)   : {dinamik['kutle_carpani']}x (Nominal: 1.00x)")
    print(f"  • Eklem Sönümleme Katsayısı       : {dinamik['eklem_sonumleme']} (Nominal: 0.15)")
    print(f"  • Motor Tork Sınır Çarpanı        : {dinamik['tork_carpani']}x (Nominal: 1.00x)")

    # -------------------------------------------------------------
    # ADIM 3: Donanım Eylem Gecikmesi Enjeksiyonu (Latency Injection)
    # -------------------------------------------------------------
    print("\n[3/4] Gerçek Donanım Gecikmesi Enjekte Ediliyor...")
    delay_inj = ActionDelayInjector(kuyruk_boyutu=3)
    eylem_1 = np.array([0.1, 0.0, 0.2])
    eylem_2 = np.array([0.2, -0.1, 0.3])
    eylem_3 = np.array([0.3, -0.2, 0.4])

    delay_inj.apply_delay(eylem_1)
    delay_inj.apply_delay(eylem_2)
    gecikmis_eylem = delay_inj.apply_delay(eylem_3, gecikme_adimi=1)
    print(f"  • Gönderilen Son Komut : {eylem_3.tolist()}")
    print(f"  • Motora Ulaşan Komut  : {gecikmis_eylem.tolist()} (Gecikmeli Eylem)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Sim2Real Teşhis Panosu Oluşturuluyor...")
    profil_raporu = Sim2RealProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "sim2real_paneli.png")

    Sim2RealGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Sim2Real Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 247 (FAZ 13): SIM2REAL TRANSFERİ VE DOMAIN RANDOMIZATION MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
