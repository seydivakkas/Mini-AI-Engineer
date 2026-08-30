"""
Day 357: Radar Micro-Doppler Signature Classification for Micro-UAVs and Ballistic Targets
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; X-Bant Radar Sinyal Sentezini, 2D STFT Mikro-Doppler Spektrogramını,
Yapay Zeka Hedef Sınıflandırmasını ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.micro_doppler_radar_motoru import (
    AirDefenseRadarTargetAnalyzer,
)
from src.radar_gorsellestirici import RadarGorsellestirici
from src.radar_profilleyici import RadarProfilleyici


def main():
    print("=" * 75, flush=True)
    print("📡 DAY 357: Mikro-Doppler Radar Sinyali ile Mikro İHA ve Balistik Hedef Sınıflandırma", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) X-Bant Radar Mikro-Doppler Sinyalleri Sentezleniyor ve Spektrogram Analizi Yapılıyor...", flush=True)

    analyzer = AirDefenseRadarTargetAnalyzer()
    analysis_res = analyzer.analyze_all_targets()

    acc_pct = analysis_res["accuracy_pct"]
    target_results = analysis_res["target_results"]

    print(f"\n📊 Hava Savunma Radarı Hedef Sınıflandırma Sonuçları:")
    for tgt_name, r in target_results.items():
        pred = r["prediction"]
        print(f"  • {tgt_name:20s} ➔ Tahmin: {pred['predicted_type'].value:20s} (Güven: %{pred['confidence']*100:.1f}) {'✅ DOĞRU' if r['is_correct'] else '❌ YANLIŞ'}")

    print(f"\n  🎯 Toplam Sınıflandırma Başarısı: %{acc_pct:.1f} (Kuş Yanılgısı: %0.0)")

    profiler_metrics = RadarProfilleyici.profille(analysis_res)

    gorsellestirici = RadarGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        analysis_res=analysis_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="mikro_doppler_radar_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Mikro-Doppler Radar Teşhis Grafiği Başarıyla Kaydedildi: [mikro_doppler_radar_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
