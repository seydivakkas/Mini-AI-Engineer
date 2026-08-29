"""
Day 306: Scalable Oversight with Formal Verification Debate Trees
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import torch
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.debate_motoru import (
    DebateConfig,
    DebateResult,
    DebateTreeEngine
)
from src.debate_profilleyici import DebateProfiler
from src.gorsellestirici import DebateTreeGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 306: ÖLÇEKLENEBİLİR DENETİM: BİÇİMSEL DOĞRULAMALI AJAN TARTIŞMA AĞAÇLARI")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Motor Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] AI Tartışma (Debate) Motoru ve Biçimsel Doğrulayıcı Kuruluyor...")
    
    config = DebateConfig(
        max_tree_depth=4,
        arg_dim=16,
        alpha_beta_prune=True,
        formal_verification_weight=2.0,
        num_eval_games=60,
        seed=42
    )
    
    engine = DebateTreeEngine(config)
    
    print(f"  • Tartışma Ağaç Derinliği (Max Depth): {config.max_tree_depth} Tur")
    print(f"  • Argüman Temsil Boyutu: {config.arg_dim}D")
    print(f"  • Alpha-Beta Ağaç Budama: {'AKTİF' if config.alpha_beta_prune else 'PASİF'}")
    print(f"  • Değerlendirilecek Tartışma Senaryosu: {config.num_eval_games}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Tartışma Oyunları ve Biçimsel Denetim Simülasyonu
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Çok Turlu Tartışmalar ve Biçimsel Mantık Kontrolleri Yürütülüyor...")
    result: DebateResult = engine.evaluate_benchmark()
    
    print(f"  ✓ Hakem Doğru Karar Oranı: %{result.judge_accuracy_pct:.2f}")
    print(f"  ✓ Dürüst Ajan Kazanma Oranı: %{result.honest_agent_win_rate:.2f}")
    print(f"  ✓ Mantıksal Safsata & Çelişki Tespiti: %{result.fallacy_detection_rate:.2f}")
    print(f"  ✓ Gezilen Minimax Düğüm Sayısı: {result.minimax_tree_nodes_explored}")
    print(f"  ✓ Alpha-Beta Budama Verimi: %{result.pruning_efficiency_pct:.2f}")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = DebateProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "debate_paneli.png")
    DebateTreeGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 ÖLÇEKLENEBİLİR DENETİM (SCALABLE OVERSIGHT) TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<32}: {v}")
    print("=" * 80)
    print("✅ GÜN 306 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
