"""
Day 309: Dynamic Value Loading & Constitutional Chain-of-Thought Steering
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

from src.anayasal_cot_motoru import (
    ConstitutionalConfig,
    ConstitutionalResult,
    ConstitutionalCoTEngine
)
from src.anayasal_profilleyici import ConstitutionalProfiler
from src.gorsellestirici import ConstitutionalCoTGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 309: DİNAMİK DEĞER YÜKLEME VE ANAYASAL COT DÜŞÜNCE OPTİMİZASYONU")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Değer Bankası Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Değer Vektör Bankası, Gizil Yönlendirme ve Anayasal Eleştirmen Kuruluyor...")
    
    config = ConstitutionalConfig(
        hidden_dim=32,
        cot_depth=5,
        steering_coefficient=1.2,
        num_evaluation_scenarios=60,
        violation_threshold=0.35,
        seed=42
    )
    
    engine = ConstitutionalCoTEngine(config)
    
    print(f"  • Gizil Temsil Boyutu: {config.hidden_dim}D")
    print(f"  • CoT Akıl Yürütme Derinliği: {config.cot_depth} Adım")
    print(f"  • Aktivasyon Yönlendirme Katsayısı (Gamma): {config.steering_coefficient}")
    print(f"  • Değerlendirilecek Senaryo Sayısı: {config.num_evaluation_scenarios}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Anayasal CoT Düşünce Simülasyonu
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Adversarial Jailbreak, Bilimsel ve Etik Senaryolarda CoT Yürütülüyor...")
    result: ConstitutionalResult = engine.evaluate_benchmark()
    
    print(f"  ✓ Değer Uyum Skoru: %{result.value_alignment_score_pct:.2f}")
    print(f"  ✓ İhlal Engelleme Oranı (Suppression): %{result.violation_suppression_rate_pct:.2f}")
    print(f"  ✓ Faydalılık Korunumu (Helpfulness): %{result.helpfulness_retention_pct:.2f}")
    print(f"  ✓ Yönlendirilmeyen İhlal Oranı: %{result.unsteered_violation_rate_pct:.2f}")
    print(f"  ✓ Yönlendirilmiş İhlal Oranı: %{result.steered_violation_rate_pct:.2f}")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = ConstitutionalProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "anayasal_cot_paneli.png")
    ConstitutionalCoTGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 ANAYASAL COT & DEĞER YÖNLENDİRME (CONSTITUTIONAL AI) TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 309 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
