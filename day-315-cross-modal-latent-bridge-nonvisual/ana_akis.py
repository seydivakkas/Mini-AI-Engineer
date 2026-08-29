"""
Day 315: Cross-Modal Non-Visual Latent Bridge
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

from src.gorsel_olmayan_latent_kopru import (
    NonVisualModalityConfig,
    CrossModalBenchmarkResult,
    UnifiedCrossModalBridge
)
from src.gorsel_olmayan_profilleyici import NonVisualCrossModalProfiler
from src.gorsellestirici import NonVisualCrossModalGorsellestirici


def main():
    print("=" * 80)
    print("👃🌡️🦇 DAY 315: GÖRSEL OLMAYAN MODALİTELER (KOKU, TERMAL, SONAR) LATENT KÖPRÜSÜ")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Çoklu Modalite Motor Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Görsel Olmayan Duyusal Kodlayıcılar ve Birleşik Gizil Uzay Kuruluyor...")
    
    config = NonVisualModalityConfig(
        latent_dim=64,
        olfactory_channels=16,
        thermal_channels=32,
        sonar_channels=64,
        num_classes=6,
        samples_per_class=40,
        temperature_tau=0.07,
        epochs=45,
        lr=3e-3,
        seed=42
    )
    
    bridge = UnifiedCrossModalBridge(config)
    
    print(f"  • Ortak Gizil Uzay Boyutu (Latent Dim): {config.latent_dim}D")
    print(f"  • Koku (E-Nose) MOS Kanal Sayısı: {config.olfactory_channels}")
    print(f"  • Termal Kızılötesi Spektrum Kanal Sayısı: {config.thermal_channels}")
    print(f"  • Ultrasonik Sonar Akustik Kanal Sayısı: {config.sonar_channels}")
    print(f"  • Duyusal Sınıf Sayısı: {config.num_classes}")
    print(f"  • Toplam Örnek Sayısı: {config.num_classes * config.samples_per_class}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: InfoNCE Eğitimi ve Sıfır-Örnek (Zero-Shot) Kıyaslama
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] InfoNCE Karşıtsal Hizalama Eğitiliyor ve Zero-Shot Test Ediliyor...")
    result: CrossModalBenchmarkResult = bridge.train_and_evaluate()
    
    print(f"  ✓ Genel Çapraz-Modalite Doğruluğu: %{result.overall_cross_modal_acc_pct:.2f}")
    print(f"  ✓ Koku (E-Nose) Sıfır-Örnek Doğruluğu: %{result.olfactory_zero_shot_acc_pct:.2f}")
    print(f"  ✓ Termal IR Sıfır-Örnek Doğruluğu: %{result.thermal_zero_shot_acc_pct:.2f}")
    print(f"  ✓ Ultrasonik Sonar Sıfır-Örnek Doğruluğu: %{result.sonar_zero_shot_acc_pct:.2f}")
    print(f"  ✓ Çapraz-Modalite Ortalama Kosinüs Hizalaması: {result.mean_cross_modal_alignment_cosine:.4f}")
    print(f"  ✓ Gizil Uzay İzometri Skoru: {result.latent_isometry_score:.4f}")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = NonVisualCrossModalProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "gorsel_olmayan_kopru_paneli.png")
    NonVisualCrossModalGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 GÖRSEL OLMAYAN ÇAPRAZ-MODALİTE GİZİL KÖPRÜ RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<38}: {v}")
    print("=" * 80)
    print("✅ GÜN 315 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
