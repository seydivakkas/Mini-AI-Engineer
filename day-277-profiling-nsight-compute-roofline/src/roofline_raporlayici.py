"""
Day 277 (FAZ 14): Nsight Compute & Roofline Raporlayıcı Modülü.
"""

from typing import Dict, Any, List
import numpy as np
from .roofline_profilleyici_motoru import NsightRooflineEngine


class RooflineRaporlayici:
    """FAZ 14 Hiyerarşik Roofline ve Nsight Donanım Raporlayıcısı."""

    @classmethod
    def basarim_profili_cikar(cls) -> Dict[str, Any]:
        """Tüm donanım hiyerarşisi ve kernel profilleri için kapsamlı rapor üretir."""
        kernel_analizleri = NsightRooflineEngine.get_standard_benchmark_suite()

        # Logaritmik Aritmetik Yoğunluk Eğrisi (0.1 - 4000 FLOP/Byte)
        intensities = np.logspace(-1, 3.6, 200)

        roof_hbm3 = [NsightRooflineEngine.calculate_attainable_performance(i, NsightRooflineEngine.HBM3_BANDWIDTH_TB_S) for i in intensities]
        roof_l2 = [NsightRooflineEngine.calculate_attainable_performance(i, NsightRooflineEngine.L2_BANDWIDTH_TB_S) for i in intensities]
        roof_sram = [NsightRooflineEngine.calculate_attainable_performance(i, NsightRooflineEngine.SRAM_BANDWIDTH_TB_S) for i in intensities]

        # Warp Scheduler Stall Dağılımı (Memory-Bound Kernel Örneği)
        warp_stalls = {
            "sebepler": [
                "1. Long Scoreboard\n(HBM3 Bellek Bekleme)",
                "2. Wait on Barrier\n(Warp Senkronizasyonu)",
                "3. MIO Throttle\n(Bellek Talimatı Sıkışması)",
                "4. Math Throttle\n(Hesaplama Boru Hattı)",
                "5. Branch / Diğer\n(Dallanma Sapması)",
            ],
            "oranlar_yuzde": [52.0, 24.0, 12.0, 8.0, 4.0],
        }

        # Donanım MFU / Verimlilik Kıyaslaması
        karsilastirma = {
            "kernel_adlari": [k["kernel_name"] for k in kernel_analizleri],
            "aritmetik_yogunluk": [k["arithmetic_intensity_flop_per_byte"] for k in kernel_analizleri],
            "ulasilan_tflops": [k["achieved_tflops"] for k in kernel_analizleri],
            "ulasilabilir_tflops": [k["attainable_tflops"] for k in kernel_analizleri],
            "donanim_verimi_yuzde": [k["hardware_efficiency_pct"] for k in kernel_analizleri],
            "darbogaz": [k["bottleneck_type"] for k in kernel_analizleri],
        }

        # İyileştirme / Hızlanma Özeti (Füzyon ile Memory-Bound -> SRAM Geçişi)
        fuzed_speedup = kernel_analizleri[2]["achieved_tflops"] / kernel_analizleri[0]["achieved_tflops"]

        return {
            "intensities": intensities,
            "roof_hbm3": roof_hbm3,
            "roof_l2": roof_l2,
            "roof_sram": roof_sram,
            "kernel_analizleri": kernel_analizleri,
            "warp_stalls": warp_stalls,
            "karsilastirma": karsilastirma,
            "fuzed_speedup": float(fuzed_speedup),
            "ridge_point": float(NsightRooflineEngine.RIDGE_POINT),
        }
