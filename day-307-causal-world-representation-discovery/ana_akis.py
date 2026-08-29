"""
Day 307: Unsupervised Latent Causal World Representation Discovery & Do-Calculus
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

from src.nedensel_dunya_motoru import (
    CausalConfig,
    CausalDiscoveryResult,
    train_and_discover_causal_graph
)
from src.nedensel_profilleyici import CausalProfiler
from src.gorsellestirici import CausalWorldGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 307: DENETİMSİZ LATENT UZAYDA NEDENSELLİK VE DO-CALCULUS TEMSİL KEŞFİ")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve SCM Ortam Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Yapısal Nedensel Model (SCM) ve NOTEARS DAG Motoru Kuruluyor...")
    
    config = CausalConfig(
        latent_dim=5,
        obs_dim=20,
        num_samples=1200,
        batch_size=64,
        lr=2e-3,
        lambda_dag=1.5,
        lambda_sparse=0.08,
        lambda_interv=0.6,
        epochs=35,
        threshold_edge=0.20,
        seed=42
    )
    
    print(f"  • Latent Nedensel Değişken Sayısı: {config.latent_dim} (z0 -> z4)")
    print(f"  • Yüksek Boyutlu Gözlem Uzayı: {config.obs_dim}D")
    print(f"  • NOTEARS DAG Kısıtlama Ağırlığı: {config.lambda_dag}")
    print(f"  • Eğitim Örnek Sayısı: {config.num_samples}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Model Eğitimi ve Nedensel Çizge Keşfi
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Sürekli DAG Optimizasyonu ve Pearl Müdahale (do) Öğrenimi Yürütülüyor...")
    result: CausalDiscoveryResult = train_and_discover_causal_graph(config)
    
    print(f"  ✓ Yapısal Hamming Uzaklığı (SHD): {result.structural_hamming_distance}")
    print(f"  ✓ Doğru Kenar Keşfi (TPR): %{result.dag_true_positive_rate_pct:.2f}")
    print(f"  ✓ Yanlış Keşif Oranı (FDR): %{result.dag_false_discovery_rate_pct:.2f}")
    print(f"  ✓ Gözlemsel Rekonstrüksiyon MSE: {result.reconstruction_mse:.4f}")
    print(f"  ✓ Müdahale (do-operator) Tahmin MSE: {result.interventional_mse:.4f}")
    print(f"  ✓ Karşı-Olgusal (Counterfactual) MSE: {result.counterfactual_mse:.4f}")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = CausalProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "nedensel_dunya_paneli.png")
    CausalWorldGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 NEDENSEL DÜNYA MODELİ (CAUSAL WORLD MODEL) TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<32}: {v}")
    print("=" * 80)
    print("✅ GÜN 307 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
