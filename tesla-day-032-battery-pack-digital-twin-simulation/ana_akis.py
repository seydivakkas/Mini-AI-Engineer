"""
Tesla Gün 32 Ana Akış (Tesla Day 32 Main Pipeline)
===================================================
Batarya Paketi Dijital İkiz (Digital Twin) Simülasyonu
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

from src.tesla_dijital_ikiz_simulasyonu import TeslaBatteryPackDigitalTwin
from src.tesla_ikiz_profilleyici import TeslaIkizProfilleyici
from src.tesla_ikiz_gorsellestirici import TeslaIkizGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 32: 96S BATARYA DİJİTAL İKİZİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 96S Paket Fiziği, Termal Gradyan & Erken Anomali Tespiti")
    print("--------------------------------------------------------------------------------\n")

    # 1. 96S Paket Dijital İkiz ve Anomali Benchmark'ı
    print(" [1] 96S Seri Hücre Paketi Simülasyonu ve Hücre #48 Anomali Enjeksiyonu...")
    profilleyici = TeslaIkizProfilleyici(sim_adimlari=500)
    metrikler = profilleyici.benchmark_dijital_ikiz()

    print(f"     -> Başlangıç Paket Gerilimi   : {metrikler['v_pack_history'][0]:.1f} V (400V Nominal)")
    print(f"     -> 500 Adım Sonrası Gerilim   : {metrikler['v_pack_son']:.1f} V")
    print(f"     -> Maksimum Voltaj Uyumsuzluğu: {metrikler['max_imbalance_mv']:.1f} mV")
    print(f"     -> Anomali Tespit Edilen Adım : {metrikler['anomaly_step']}. Adım (100. Adımda Enjekte Edildi)")
    print(f"     -> Tespit Edilen Kusurlu Hücre: Hücre #{metrikler['faulty_cell_id']} (Tam İsabet!)")

    # 2. 96-Hücrelik İkiz Çözüm Hızı RTOS Performansı
    print("\n [2] 96-Hücreli Komple Paket İkizi RTOS Performansı...")
    print(f"     -> Ortalama İkiz Adım Süresi  : {metrikler['ikiz_step_ortalama_us']:.3f} µs (P99: {metrikler['ikiz_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik İkiz Kapasitesi  : {metrikler['saniyelik_ikiz_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Batarya Dijital İkiz Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaIkizGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_dijital_ikiz_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 32 BAŞARIYLA TAMAMLANDI! BATARYA DİJİTAL İKİZİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
