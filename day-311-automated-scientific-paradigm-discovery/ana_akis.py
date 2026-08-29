"""
Day 311: Automated Scientific Theory & Paradigm Discovery Engine
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.bilimsel_kesif_motoru import (
    ScientificDiscoveryConfig,
    ScientificDiscoveryResult,
    AutomatedScientificDiscoveryEngine
)
from src.bilimsel_kesif_profilleyici import ScientificDiscoveryProfiler
from src.gorsellestirici import ScientificDiscoveryGorsellestirici


def main():
    print("=" * 80)
    print("🔬 DAY 311: OTONOM BİLİMSEL TEORİ VE PARADİGMA KEŞİF MOTORU (SINDy & AI SCIENTIST)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Sembolik Kütüphane Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Sembolik Fonksiyon Aday Kütüphanesi Theta(X) ve SINDy Modülü Kuruluyor...")
    
    config = ScientificDiscoveryConfig(
        poly_order=3,
        include_trig=True,
        sparsity_threshold=0.08,
        ridge_alpha=1e-4,
        noise_level=0.01,
        time_steps=500,
        dt=0.02,
        seed=42
    )
    
    engine = AutomatedScientificDiscoveryEngine(config)
    
    print(f"  • Polinom Derecesi (Poly Order): {config.poly_order}")
    print(f"  • Trigonometrik Terimler Dahil: {config.include_trig}")
    print(f"  • Seyreklik Eşik Değeri (Lambda): {config.sparsity_threshold}")
    print(f"  • Sensör Gürültü Seviyesi (Sigma): {config.noise_level}")
    print(f"  • Gözlem Zaman Adımı Sayısı: {config.time_steps} (dt = {config.dt}s)")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Dinamik Veri Üretimi ve Seyrek Denklem Keşfi
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Lorenz Kaotik Dinamiğinden Sembolik Fizik Yasaları Keşfediliyor...")
    result: ScientificDiscoveryResult = engine.discover_laws()
    
    print(f"  ✓ Denklem Geri Kazanım Kesinliği: %{result.equation_recovery_precision_pct:.2f}")
    print(f"  ✓ Parametre Bağıl Hatası: %{result.avg_parameter_relative_error_pct:.2f}")
    print(f"  ✓ Dağılım Dışı (OOD) Genelleme R²: {result.ood_extrapolation_r2:.4f}")
    print(f"  ✓ Model Yalınlığı (Parsimony BIC): {result.parsimony_bic_score:.2f}")
    
    print("\n  📐 Keşfedilen Diferansiyel Denklemler:")
    for var, eq in result.discovered_equations.items():
        print(f"    • {var} = {eq}")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = ScientificDiscoveryProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "bilimsel_kesif_paneli.png")
    ScientificDiscoveryGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 OTONOM BİLİMSEL TEORİ KEŞFİ TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 311 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
