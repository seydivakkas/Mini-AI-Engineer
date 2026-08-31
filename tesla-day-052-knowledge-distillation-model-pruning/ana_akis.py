"""
Tesla Gün 52 Ana Akış (Tesla Day 52 Main Pipeline)
===================================================
Model Damıtma (Knowledge Distillation) ve Yapısal Budama
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

from src.tesla_model_damitma_ve_budama import TeslaKnowledgeDistiller
from src.tesla_damitma_profilleyici import TeslaDamitmaProfilleyici
from src.tesla_damitma_gorsellestirici import TeslaDamitmaGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 52: MODEL DAMITMA (KD) VE YAPISAL BUDAMA 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Teacher-Student Transfer, Sıcaklık T=4, L1-Norm Budama & FLOPs")
    print("--------------------------------------------------------------------------------\n")

    # 1. Damıtma ve Budama Benchmark'ı
    print(" [1] Teacher-Student Bilgi Damıtma ve L1-Norm Kanal Budama Çözümleniyor...")
    profilleyici = TeslaDamitmaProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_distillation_and_pruning()

    print(f"     -> Toplam Damıtma Kaybı     : {metrikler['total_loss']:.4f} (Soft KD: {metrikler['loss_soft_kd']:.4f})")
    print(f"     -> KL-Divergence Farkı      : {metrikler['kl_div']:.4f}")
    print(f"     -> Yapısal Budama Oranı     : %{metrikler['sparsity_pct']:.1f} ({metrikler['pruned_channels_count']} Kanal Kapatıldı)")
    print(f"     -> Aktif Kalan Kanal Sayısı : {metrikler['active_channels_count']} / 64 Kanal")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Model Damıtma ve Budama RTOS Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['distill_step_ortalama_us']:.3f} µs (P99: {metrikler['distill_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Adım Hacmi     : {metrikler['saniyelik_damitma_adimi']:,} Adım/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Model Damıtma ve Budama Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaDamitmaGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_distillation_pruning_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 52 BAŞARIYLA TAMAMLANDI! MODEL DAMITMA VE BUDAMA DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
