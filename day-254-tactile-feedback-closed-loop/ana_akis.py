"""
Day 254: Kapalı Çevrim Dokunsal Geri Bildirim Kontrolü (Tactile Feedback Control) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.tactile_feedback_motoru import (
    TactileSlipDetector,
    AdaptiveStiffnessEstimator,
    ClosedLoopTactileController,
)
from src.tactile_feedback_profilleyici import TactileFeedbackProfilleyici
from src.gorsellestirici import TactileFeedbackGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 254 (FAZ 13): KAPALI ÇEVRİM DOKUNSAL GERİ BİLDİRİM KONTROLÜ (CLOSED-LOOP TACTILE FEEDBACK)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Dokunsal Kayma ve Sertlik Modüllerinin Kurulması
    # -------------------------------------------------------------
    print("\n[1/4] 1000 Hz Dokunsal Kayma Dedektörü ve Sertlik Kestirici Başlatılıyor...")
    mu_s = 0.55
    stiffness_info = AdaptiveStiffnessEstimator.estimate_stiffness(delta_f_N=0.6, delta_x_mm=2.5)

    print(f"  • Statik Sürtünme Katsayısı (mu_s) : {mu_s}")
    print(f"  • Kestirilen Sertlik (k_est)        : {stiffness_info['sertlik_k_N_mm']} N/mm")
    print(f"  • Tespit Edilen Nesne Sınıfı        : {stiffness_info['nesne_sinifi']}")
    print(f"  • Maksimum Güvenli Kuvvet Tavanı    : {stiffness_info['maksimum_guvenli_kuvvet_N']} N")

    # -------------------------------------------------------------
    # ADIM 2: Kapalı Çevrim Dokunsal Kontrolcünün Başlatılması
    # -------------------------------------------------------------
    print("\n[2/4] Kapalı Çevrim Değişken Empedanslı Tutuş Kontrolcüsü Devrede...")
    controller = ClosedLoopTactileController(mu_s=mu_s)
    controller.f_normal = 0.8  # Başlangıç tutuşu 0.8N
    print(f"  • Başlangıç Normal Kuvvet (F_n)     : {controller.f_normal} N")

    # -------------------------------------------------------------
    # ADIM 3: Dinamik Kayma ve Hassas Nesne Tutuş Simülasyonu
    # -------------------------------------------------------------
    print("\n[3/4] Kırılgan Nesne Tutuşu ve Ani Kayma Bozucusu Simülasyonu (4 Adım Döngüsü)...")
    senaryo = [
        {"ft": 0.2, "dx": 2.0, "df": 0.5, "vib": None, "aciklama": "Normal Kararlı Tutuş"},
        {"ft": 0.7, "dx": 0.3, "df": 0.2, "vib": np.random.randn(40)*0.5, "aciklama": "Ani Yerçekimi / Kayma Başlangıcı"},
        {"ft": 0.4, "dx": 0.1, "df": 0.1, "vib": None, "aciklama": "Kompanzasyon Sonrası Denge"},
        {"ft": 0.3, "dx": 0.0, "df": 0.0, "vib": None, "aciklama": "Kararlı ve Güvenli Tutuş"},
    ]

    for i, adim in enumerate(senaryo, 1):
        res = controller.step_control(
            f_tangential=adim["ft"],
            delta_x_gripper_mm=adim["dx"],
            delta_f_sensor_N=adim["df"],
            vib_signal=adim["vib"],
        )
        durum = res["durum"]
        fn = res["uygulanan_f_normal_N"]
        kayma = res["kayma_bilgisi"]["kayma_tehlikesi_var_mi"]
        ezilme = res["ezilme_riski_var_mi"]
        print(f"  [Adım {i}] {adim['aciklama']} -> Durum: {durum} | F_n: {fn}N | Kayma: {kayma} | Ezilme Tehlikesi: {ezilme}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Dokunsal Geri Bildirim Teşhis Panosu Oluşturuluyor...")
    profil_raporu = TactileFeedbackProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "tactile_feedback_paneli.png")

    TactileFeedbackGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Dokunsal Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 254 (FAZ 13): KAPALI ÇEVRİM DOKUNSAL GERİ BİLDİRİM KONTROL MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
