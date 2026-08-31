"""
Tesla Gün 48 Ana Akış (Tesla Day 48 Main Pipeline)
===================================================
VectorLaneNet: Yol Çizgisi, Şerit Sınırları ve Kavşak Topolojisi
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

from src.tesla_vector_lanenet_graf_topolojisi import TeslaVectorLaneNet
from src.tesla_lanenet_profilleyici import TeslaLaneNetProfilleyici
from src.tesla_lanenet_gorsellestirici import TeslaLaneNetGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 48: VectorLaneNet VE YOL GRAF TOPOLOJİSİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 3. Derece Şerit Polinomları, Yönlendirilmiş Graf (DAG) & Eğrilik")
    print("--------------------------------------------------------------------------------\n")

    # 1. VectorLaneNet Benchmark'ı
    print(" [1] Kavşak Yönlendirilmiş Grafı ve Şerit Polinomları Çözümleniyor...")
    profilleyici = TeslaLaneNetProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_vector_lanenet()

    g = metrikler["graph"]
    print(f"     -> Tanımlanan Şerit Düğümü   : {g['node_count']} Adet (Sol, Sağ, Düz, Dönüşler)")
    print(f"     -> 10m İlerideki Şerit Eğriliği: {metrikler['kappa_10m']:.6f} 1/m")
    print(f"     -> Sol Şeritten Geçiş Seçenekleri: Şerit ID'leri {metrikler['legal_next_lanes_0']}")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] VectorLaneNet RTOS Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi     : {metrikler['lanenet_step_ortalama_us']:.3f} µs (P99: {metrikler['lanenet_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Şerit Graf Hacmi: {metrikler['saniyelik_lanenet_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD VectorLaneNet Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaLaneNetGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_vector_lanenet_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi    : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 48 BAŞARIYLA TAMAMLANDI! VectorLaneNet GRAF TOPOLOJİSİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
