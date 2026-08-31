"""
Tesla Spatiotemporal BEV Transformer Profilleyici Modülü
========================================================
Bu modül; Mekansal Çapraz Dikkat, Ego-Motion Warp ve Zamansal Bellek Füzyonunun
gecikmesini, oklüzyon (geçici görünmezlik) dayanıklılığını ve doluluk haritasını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_spatiotemporal_bev_transformer import TeslaSpatiotemporalBEVTransformer


class TeslaTransformerProfilleyici:
    """
    BEV Transformer Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_spatiotemporal_transformer(self) -> Dict[str, Any]:
        transformer = TeslaSpatiotemporalBEVTransformer(bev_h=50, bev_w=50, feature_dim=32)

        # 8 Kamera İçin Sentetik 2D Öznitelik Tensörleri
        cam_names = [
            "Front_Main", "Front_Narrow", "Front_Wide",
            "Left_Pillar", "Right_Pillar", "Left_Repeater", "Right_Repeater", "Rear_View"
        ]

        gecikmeler_us: List[float] = []
        occlusion_memory_probs = []

        last_out = {}
        for frame in range(self.iterations):
            # 20. karede önde bir araç algılanır (BEV gridde [30, 25] hücresinde sinyal)
            # 21-25. karelerde kameralarda oklüzyon olur (öznitelik sıfırlanır)
            cam_feats = {}
            for name in cam_names:
                f = np.zeros((50, 50, 32))
                if frame < 20:
                    f[30, 25, :] = 2.5  # Öndeki araç sinyali
                cam_feats[name] = f

            dx = 1.0  # 1 metre ileri hareket (36 km/h @ 10 Hz)
            dy = 0.0
            dyaw = 0.0

            t0 = time.perf_counter_ns()
            out = transformer.step(cam_feats, dx, dy, dyaw)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

            # Aracın olduğu tahmin edilen hücrenin olasılığı
            # İleri hareketle hücre indeksi geriye kayar (30 - 1 = 29 vb.)
            target_idx = max(0, 30 - frame)
            occlusion_memory_probs.append(float(out["occupancy_prob"][target_idx, 25]))
            last_out = out

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))

        return {
            "transformer_step_ortalama_us": t_avg_us,
            "transformer_step_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_bev_adim": int(1e6 / max(t_avg_us, 1e-4)),
            "occlusion_memory_probs": occlusion_memory_probs,
            "final_occupancy_prob": last_out.get("occupancy_prob", np.zeros((50, 50))),
            "gecikmeler": gecikmeler_us[:200]
        }
