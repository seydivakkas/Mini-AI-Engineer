"""
Day 303: Open-Ended Quality-Diversity Algorithms (MAP-Elites & POET)
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.map_elites_poet_motoru import (
    POETEngine,
    MAPElitesEngine,
    QDConfig,
    QDResult
)
from src.poet_profilleyici import POETProfiler
from src.gorsellestirici import POETGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 303: UCU AÇIK EVRİMSEL KALİTE-ÇEŞİTLİLİK (MAP-ELITES & POET)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Kök Ortam Hazırlığı
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Quality-Diversity Parametreleri ve Kök Ortam Başlatılıyor...")
    
    config = QDConfig(
        grid_dim=16,             # 16x16 = 256 Davranışsal Niche
        num_iterations=40,
        batch_size=20,
        mutation_sigma=0.08,
        poet_max_envs=6,
        transfer_interval=4,
        seed=42
    )
    
    print(f"  • MAP-Elites Izgara Boyutu: {config.grid_dim}x{config.grid_dim} ({config.grid_dim**2} Niche)")
    print(f"  • İterasyon Sayısı: {config.num_iterations} | Batch Boyutu: {config.batch_size}")
    print(f"  • Maksimum POET Ortam Sayısı: {config.poet_max_envs}")
    print(f"  • Politika Çapraz Transfer Aralığı: Her {config.transfer_interval} iterasyonda bir")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: MAP-Elites & POET Eş-Zamanlı Evrim Motorunu Çalıştırma
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] POET ve MAP-Elites Ucu Açık Eş-Zamanlı Evrimi Başlatılıyor...")
    poet_engine = POETEngine(config, obs_dim=8, act_dim=2)
    result: QDResult = poet_engine.run_poet()
    
    print(f"  ✓ Evrimsel Döngü Tamamlandı!")
    print(f"  ✓ Toplam Yapılan Simülasyon / Değerlendirme: {result.total_evaluations}")
    print(f"  ✓ Oluşturulan Aktif Ortam Sayısı: {len(result.active_envs)}")
    print(f"  ✓ En İyi Birey Uygunluğu (Fitness): {result.best_individual.fitness:.2f}")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = POETProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "poet_qd_paneli.png")
    POETGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 QUALITY-DIVERSITY & POET TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<32}: {v}")
    print("=" * 80)
    print("✅ GÜN 303 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
