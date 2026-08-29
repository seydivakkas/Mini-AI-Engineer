"""
Day 310: Diffusion-Based Latent Planner & Trajectory Sampling Engine
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

from src.difuzyon_planlayici_motoru import (
    DiffusionPlannerConfig,
    DiffusionPlannerResult,
    GoalConditionedDiffusionPlanner
)
from src.difuzyon_profilleyici import DiffusionPlannerProfiler
from src.gorsellestirici import DiffusionPlannerGorsellestirici


def main():
    print("=" * 80)
    print("🚀 DAY 310: DİFÜZYON TABANLI LATENT DÜŞÜNCE PLANLAMASI VE YÖRÜNGE ÖRNEKLEME")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Model Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] 1D Zamansal UNet, Gürültü Zamanlayıcı (Scheduler) ve Ortam Kuruluyor...")
    
    config = DiffusionPlannerConfig(
        trajectory_len=32,
        state_dim=4,
        num_diffusion_steps=40,
        guidance_scale=2.5,
        num_eval_trajectories=50,
        learning_rate=1e-3,
        seed=42
    )
    
    planner = GoalConditionedDiffusionPlanner(config)
    
    print(f"  • Yörünge Ufku (Horizon H): {config.trajectory_len} Adım")
    print(f"  • Durum/Gizil Uzay Boyutu: {config.state_dim}D (x, y, vx, vy)")
    print(f"  • Toplam Difüzyon Adım Sayısı (T): {config.num_diffusion_steps}")
    print(f"  • Sınıflandırıcısız Yönlendirme (CFG w): {config.guidance_scale}")
    print(f"  • Değerlendirilecek Yörünge Sayısı: {config.num_eval_trajectories}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Ters Difüzyon Örnekleme ve Kapsamlı Değerlendirme
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Skor Tabanlı Ters Difüzyon ile Yörüngeler Örnekleniyor...")
    result: DiffusionPlannerResult = planner.evaluate_benchmark()
    
    print(f"  ✓ Hedefe Ulaşma Oranı (Reachability): %{result.goal_reachability_rate_pct:.2f}")
    print(f"  ✓ Engelden Kaçınma Oranı (Avoidance): %{result.obstacle_avoidance_rate_pct:.2f}")
    print(f"  ✓ Yörünge Pürüzsüzlük Skoru: {result.trajectory_smoothness_score:.2f}/100")
    print(f"  ✓ DDIM Hızlandırma Katsayısı: {result.ddim_speedup_factor:.1f}x")
    print(f"  ✓ Ortalama Yörünge Uzunluğu: {result.avg_trajectory_length:.2f} birim")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = DiffusionPlannerProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "difuzyon_planlayici_paneli.png")
    DiffusionPlannerGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 DİFÜZYON TABANLI LATENT PLANLAYICI TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 310 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
