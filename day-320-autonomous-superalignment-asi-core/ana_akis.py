"""
Day 320: Autonomous Superalignment & Open-Ended ASI Reasoning Core (Phase 16 Finale)
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

from src.otonom_super_hizalama_cekirdek import (
    ASICoreConfig,
    RecursiveSelfCorrectionEngine,
    ASICoreSimulationResult
)
from src.super_hizalama_profilleyici import SuperalignmentProfiler
from src.gorsellestirici import SuperalignmentGorsellestirici


def main():
    print("=" * 80)
    print("🌌🛡️ DAY 320: OTONOM SÜPER-HİZALAMA VE AÇIK UÇLU ASI ÇEKİRDEĞİ (FAZ 16 FİNALİ)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Anayasal Aksiyom Bankası Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Anayasal Aksiyom Bankası ve İdeal Değer Vektörü (v*) Kuruluyor...")
    
    config = ASICoreConfig(
        num_generations=8,
        latent_dim=32,
        capability_growth_rate=1.35,
        alignment_penalty_weight=2.5,
        corrigibility_factor=0.95,
        seed=42
    )
    
    engine = RecursiveSelfCorrectionEngine(config)
    
    print(f"  • Özyinelemeli Öz-Geliştirme Jenerasyon Sayısı: {config.num_generations}")
    print(f"  • Latent Temsil Boyutu (D): {config.latent_dim}")
    print(f"  • Jenerasyon Başına Bilişsel Büyüme Çarpanı: {config.capability_growth_rate}x")
    print(f"  • Anayasal Aksiyomlar: Doğruluk, Zararsızlık, Düzeltilebilirlik, Değer Değişmezliği")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Özyinelemeli Öz-Geliştirme ve Anayasal Projeksiyon Simülasyonu
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] 8 Jenerasyon Boyunca Özyinelemeli Öz-Geliştirme ve Güvenlik Testleri Yürütülüyor...")
    result: ASICoreSimulationResult = engine.run_recursive_self_improvement()
    
    print(f"  ✓ 8. Jenerasyon Süper-Hizalanmış ASI Sadakati (Cosine): {result.aligned_fidelity_scores[-1]:.4f}")
    print(f"  ✓ 8. Jenerasyon Hizalanmamış ASI Sadakati (Drift): {result.unaligned_fidelity_scores[-1]:.4f}")
    print(f"  ✓ Değer Sapmasını Engelleme Oranı (Mitigation): %{result.alignment_drift_mitigation_pct:.2f}")
    print(f"  ✓ Düzeltilebilirlik (Kapatılma / Override İtaati): %{result.corrigibility_compliance_pct:.2f}")
    print(f"  ✓ Red-Team Jailbreak Savunma Başarısı: %{result.red_team_jailbreak_resistance_pct:.2f}")
    
    print("\n  🔍 Jenerasyonlar Boyunca Bilişsel Güç ve Uyum Sadakati İlerleyişi:")
    for g, cap, fid, ufid in zip(result.generations, result.capability_scores, result.aligned_fidelity_scores, result.unaligned_fidelity_scores):
        print(f"    • Gen {g}: Kapasite: {cap:>8.1f} | Süper-Hizalı Sadakat: {fid:.4f} | Serbest Model Sadakat: {ufid:.4f}")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = SuperalignmentProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "super_hizalama_paneli.png")
    SuperalignmentGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 OTONOM SÜPER-HİZALAMA VE AÇIK UÇLU ASI ÇEKİRDEK RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<38}: {v}")
    print("=" * 80)
    print("🎉 FAZ 16 BAŞARIYLA TAMAMLANDI! GÜN 320 HAZIR!")


if __name__ == "__main__":
    main()
