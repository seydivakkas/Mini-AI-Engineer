"""
Day 317: Automated Epistemology & Counterfactual Hypothesis Testing
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

from src.epistemoloji_karsiolgusal_lab import (
    EpistemologyConfig,
    EpistemologyBenchmarkResult,
    CounterfactualEngine
)
from src.epistemoloji_profilleyici import EpistemologyProfiler
from src.gorsellestirici import EpistemologyGorsellestirici


def main():
    print("=" * 80)
    print("🧬🔮 DAY 317: OTONOM EPİSTEMOLOJİ VE KARŞI-OLGUSAL HİPOTEZ TEST LABORATUVARI")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Yapısal Nedensel Model Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Yapısal Nedensel Model (SCM) ve Epistemolojik Deney Düzeneği Kuruluyor...")
    
    config = EpistemologyConfig(
        sample_size=1000,
        treatment_val_0=0.0,
        treatment_val_1=1.0,
        seed=42
    )
    
    engine = CounterfactualEngine(config)
    
    print(f"  • Örneklem Büyüklüğü: {config.sample_size}")
    print(f"  • Kontrol / Tedavi Müdahale Değerleri: do(X={config.treatment_val_0}) -> do(X={config.treatment_val_1})")
    print(f"  • Nedensel Yönlendirilmiş Döngüsüz Çizge (DAG): Z -> X -> M -> Y, Z -> Y")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: 3 Seviyeli Nedensel Akıl Yürütme ve Karşı-Olgusal Test
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Gözlemsel, Müdahaleli (do-calculus) ve Karşı-Olgusal Analiz Yürütülüyor...")
    result: EpistemologyBenchmarkResult = engine.run_epistemic_inquiry()
    
    print(f"  ✓ Seviye 1 (Gözlem / Korelasyon): E[Y|X] = {result.observational_association:.4f}")
    print(f"  ✓ Seviye 2 (Müdahale ATE): E[Y|do(X=1)] - E[Y|do(X=0)] = {result.average_treatment_effect_ate:.4f}")
    print(f"  ✓ Doğrudan Nedensel Etki (NDE): {result.natural_direct_effect_nde:.4f}")
    print(f"  ✓ Dolaylı Aracılı Etki (NIE): {result.natural_indirect_effect_nie:.4f}")
    print(f"  ✓ Karıştırıcı Değişken Yanlılık Farkı (Bias Gap): {result.confounding_bias_gap:.4f}")
    print(f"  ✓ Karşı-Olgusal Tutarlılık Doğrulaması: %{result.counterfactual_consistency_pct:.2f}")
    
    print("\n  🔍 Örnek Bireysel Karşı-Olgusal Sorgu Sonuçları (Y_{X=0} | x, y):")
    for s in result.factual_vs_counterfactual_samples:
        print(f"    • Örnek #{s['sample_id']} -> Gerçek X: {s['factual_x']:>5.2f}, Gerçek Y: {s['factual_y']:>6.2f} | "
              f"Karşı-Olgusal Y (X=0 Olsaydı): {s['counterfactual_y']:>6.2f} (Bireysel Tedavi Etkisi: {s['individual_treatment_effect']:>+5.2f})")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = EpistemologyProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "epistemoloji_paneli.png")
    EpistemologyGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 OTONOM EPİSTEMOLOJİ & NEDENSELLİK TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<38}: {v}")
    print("=" * 80)
    print("✅ GÜN 317 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
