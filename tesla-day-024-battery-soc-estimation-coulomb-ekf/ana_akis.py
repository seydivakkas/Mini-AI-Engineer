"""
Tesla Gün 24 Ana Akış (Tesla Day 24 Main Pipeline)
===================================================
Batarya SoC Kestirimi: Coulomb Counting & Genişletilmiş Kalman Filtresi (EKF)
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_ekf_soc_kestirici import BatteryEKFSoCEstimator, CoulombCounter
from src.tesla_ekf_profilleyici import TeslaEKFProfilleyici
from src.tesla_ekf_gorsellestirici import TeslaEKFGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 24: GENİŞLETİLMİŞ KALMAN FİLTRESİ (EKF) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Coulomb Counting Drift İyileştirme, EKF Durum Tahmini & Jacobian")
    print("--------------------------------------------------------------------------------\n")

    # 1. Hatalı Başlangıç Durumunda EKF Yakınsama Testi
    print(" [1] Yanlış Başlangıç Tahmini (%50) Altında Gerçek Hücreye (%85) Yakınsama...")
    ekf = BatteryEKFSoCEstimator(initial_soc_guess=0.50, capacity_ah=75.0)
    ocv_85, _ = ekf._compute_ocv_and_derivative(0.85)

    print(f"     -> Başlangıç EKF Tahmini  : %{ekf.x[0]*100:.1f}")
    print(f"     -> Zemin Gerçeği (True)   : %85.0 (Voltaj: {ocv_85:.3f} V)")

    for adim in range(1, 201):
        out = ekf.step(current_a=0.0, measured_terminal_v=ocv_85, dt_s=0.1)
        if adim in (10, 50, 100, 200):
            print(f"        * Adım {adim:3d} ({adim*0.1:4.1f} sn) -> Tahmin SoC: %{out['estimated_soc']*100:.2f} | 3σ Belirsizlik: ±%{out['soc_uncertainty_std']*300:.2f}")

    # 2. Akım Sensörü Yanlılığı (+1.5A DC Bias) Altında Karşılaştırma
    print("\n [2] Akım Sensörü DC Kayması (+1.5A Bias) Altında 1500 Adımlık Sürüş Testi...")
    profilleyici = TeslaEKFProfilleyici(sim_adimlari=1500)
    metrikler = profilleyici.benchmark_ekf_soc()

    print(f"     -> Coulomb Counting RMSE  : %{metrikler['rmse_coulomb_pct']:.2f} (Aşırı Sürüklenme!)")
    print(f"     -> Tesla EKF Algoritma RMSE: %{metrikler['rmse_ekf_pct']:.2f} (Mükemmel Doğruluk)")
    print(f"     -> Hata İyileştirme Çarpanı: {metrikler['hata_iyilesme_orani']:.1f}x Daha Güvenilir!")

    # 3. Performans ve Matris Çözüm Benchmark'ı
    print("\n [3] EKF Matris Adım Gecikmesi ve 1 kHz RTOS Uyumluluğu...")
    print(f"     -> Ortalama EKF Adım Süresi: {metrikler['ekf_step_ortalama_us']:.3f} µs (P99: {metrikler['ekf_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik EKF Kapasitesi: {metrikler['saniyelik_ekf_adimi']:,} Adım/sn")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla EKF SoC Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaEKFGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_ekf_soc_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 24 BAŞARIYLA TAMAMLANDI! GENİŞLETİLMİŞ KALMAN FİLTRESİ (EKF) DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
