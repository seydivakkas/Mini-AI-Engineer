"""
Day 351: Satellite Constellation Edge AI for Real-Time Wildfire & Thermal Anomaly Detection
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Çok Bantlı Uydu Gözlemini, Kuantize Edge AI Yangın Segmentasyonunu,
FRP Termal Güç Hesabını ve 6-Panelli Uydu Teşhis Grafiğini çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.satellite_wildfire_edge_motoru import (
    MultispectralEarthSimulator,
    OnBoardWildfireEdgeDetector,
    SatelliteConstellationNetwork,
)
from src.wildfire_gorsellestirici import WildfireGorsellestirici
from src.wildfire_profilleyici import WildfireProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🛰️ DAY 351: Uydu Takımyıldızı Edge AI: Gerçek Zamanlı Orman Yangını ve Tehdit Tespiti", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. 4-Bantlı Multispektral Dünya Yüzeyi Simülasyonu
    sim = MultispectralEarthSimulator(grid_size=64)
    multispectral, fire_mask_true = sim.generate_multispectral_tile(has_wildfire=True)

    # 2. Uydu Üzeri (On-Board) Edge AI Yangın Algılama
    edge_detector = OnBoardWildfireEdgeDetector()
    constellation = SatelliteConstellationNetwork(num_sats=6)

    print("\n📌 1) LEO Küp Uydu Üzerinde (On-Board) Çok Bantlı Edge AI Segmentasyonu Başlatılıyor...", flush=True)

    t0 = time.perf_counter()
    res = edge_detector.detect_wildfire(multispectral)
    infer_time_ms = (time.perf_counter() - t0) * 1000.0

    pred_mask = res["pred_mask"]
    nbr_map = res["nbr_map"]
    alert = res["alert_payload"]

    # 3. Alarm İletim Gecikmesi
    downlink_latency_ms = constellation.route_alert_to_ground(alert)

    # Metrikler
    profiler_metrics = WildfireProfilleyici.profille(
        fire_mask_true=fire_mask_true,
        fire_mask_pred=pred_mask,
        total_frp_mw=alert["total_frp_mw"]
    )

    print(f"\n📊 Uydu Edge AI Yangın Algılama Performans Sonuçları:")
    print(f"  • Tespit Duyarlılığı (Recall):    %{profiler_metrics['recall_score']:.2f} (> 95% Kriteri)")
    print(f"  • Kesinlik (Precision):           %{profiler_metrics['precision_score']:.2f}")
    print(f"  • Piksel Kesişim Başarısı (IoU):  %{profiler_metrics['iou_score']*100:.2f}")
    print(f"  • Yangın Işınım Gücü (FRP):       {alert['total_frp_mw']:.2f} MegaWatt (Kritik Yangın)")
    print(f"  • Uydu Üzeri AI Çıkarım Süresi:   {infer_time_ms:.4f} ms (< 15 ms Edge Kriteri)")
    print(f"  • Toplam Yere Uyarı Gecikmesi:    {downlink_latency_ms:.1f} ms (Gerçek Zamanlı Erken Uyarı)")

    print("\n📡 Üretilen Uydu Erken Uyarı JSON Paketi (Geo-Alert):")
    print(f"  • Yangın Durumu  : {alert['satellite_edge_alert']}")
    print(f"  • Yanan Piksel   : {alert['fire_pixel_count']} Adet")
    print(f"  • Ortalama Sıcaklık: {alert['mean_fire_temp_k']:.1f} K")
    print(f"  • Güven Seviyesi : %{alert['confidence']*100:.1f}")

    gorsellestirici = WildfireGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        multispectral=multispectral,
        fire_mask_true=fire_mask_true,
        fire_mask_pred=pred_mask,
        nbr_map=nbr_map,
        alert_payload=alert,
        profiler_metrics=profiler_metrics,
        dosya_adi="uydu_yangin_edge_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Uydu Edge AI Teşhis Grafiği Başarıyla Kaydedildi: [uydu_yangin_edge_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
