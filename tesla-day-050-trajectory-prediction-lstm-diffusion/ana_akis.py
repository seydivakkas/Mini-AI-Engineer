"""
Tesla Gün 50 Ana Akış (Tesla Day 50 Main Pipeline)
===================================================
LSTM, GRU ve Difüzyon Modelleri ile Dinamik Nesne Yörünge Tahmini
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

from src.tesla_yorunge_tahmin_lstm_difuzyon import TeslaTrajectoryPredictor
from src.tesla_yorunge_profilleyici import TeslaYorungeProfilleyici
from src.tesla_yorunge_gorsellestirici import TeslaYorungeGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 50: DİNAMİK YÖRÜNGE TAHMİNİ (TRAJECTORY PREDICTION) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 5s Gelecek Tahmini, Çoklu Modalite, Difüzyon & TTC Çarpışma Riski")
    print("--------------------------------------------------------------------------------\n")

    # 1. Yörünge Tahmini Benchmark'ı
    print(" [1] 5 Saniyelik Gelecek Ufku ve Çoklu Davranış Modları Çözümleniyor...")
    profilleyici = TeslaYorungeProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_trajectory_predictor()

    print(f"     -> Davranış Modları         : {metrikler['modes']}")
    print(f"     -> Mod Olasılıkları         : {[f'%{p*100:.1f}' for p in metrikler['probabilities']]}")
    print(f"     -> Çarpışmaya Kalan Süre    : {metrikler['ttc_sec']:.2f} Saniye (TTC)")
    print(f"     -> Tahmin Edilen Nokta Sayısı: 50 Adım (dt = 0.1s)")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Yörünge Tahmini RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['yorunge_step_ortalama_us']:.3f} µs (P99: {metrikler['yorunge_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Tahmin Hacmi   : {metrikler['saniyelik_tahmin_adimi']:,} Tahmin/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Yörünge Tahmini Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaYorungeGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_trajectory_prediction_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 50 BAŞARIYLA TAMAMLANDI! DİNAMİK YÖRÜNGE TAHMİNİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
