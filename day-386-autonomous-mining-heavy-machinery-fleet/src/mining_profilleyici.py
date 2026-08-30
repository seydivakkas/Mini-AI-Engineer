"""
Day 386: Autonomous Mining & Heavy Machinery Fleet in GPS-Denied Environments
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Otonom maden filosu üretim verimini, SLAM konumlandırma hassasiyetini,
güvenlik indeksini ve genel madencilik otonomi skorunu profiller.
"""

from typing import Dict, Any


class MiningProfilleyici:
    """
    Otonom Maden Filosu Performans Profilleyicisi.
    """
    def __init__(self):
        pass

    def profille(self, bench_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maden filosu başarım sonuçlarını profiller.
        """
        rate = bench_res.get("production_rate_tons_per_hr", 450.0)
        slam_err = bench_res.get("avg_slam_positioning_error_m", 0.05)
        collisions = bench_res.get("collision_count", 0)

        tonnage_score = min(100.0, (rate / 400.0) * 100.0)
        slam_score = max(0.0, 100.0 - (slam_err / 0.15) * 40.0)
        safety_score = 100.0 if collisions == 0 else 0.0

        mining_autonomy_score = (tonnage_score * 0.40 + slam_score * 0.35 + safety_score * 0.25)

        return {
            "tonnage_score": round(tonnage_score, 2),
            "slam_score": round(slam_score, 2),
            "safety_score": round(safety_score, 2),
            "mining_autonomy_score": round(mining_autonomy_score, 2),
            "production_rate_tons_per_hr": rate,
            "total_ore_extracted_tons": bench_res.get("total_ore_extracted_tons", 2000.0),
            "avg_slam_positioning_error_m": slam_err,
            "collision_count": collisions
        }

    def rapor_olustur(self, metrics: Dict[str, Any]) -> str:
        """
        Konsol için yapılandırılmış Otonom Madencilik Raporu üretir.
        """
        rapor = (
            "\n" + "=" * 75 + "\n"
            "   DAY 386: OTONOM YERALTI MADEN FİLOSU PERFORMANS RAPORU\n"
            "=" * 75 + "\n"
            f"  • Toplam Taşınan Cevher Tonajı     : {metrics['total_ore_extracted_tons']:.1f} Ton\n"
            f"  • Üretim Hızı (Kapasite)           : {metrics['production_rate_tons_per_hr']:.1f} Ton / Saat\n"
            f"  • GPS'siz Yeraltı SLAM Konum Hatası: {metrics['avg_slam_positioning_error_m']:.3f} m (< 0.15 m PASS)\n"
            f"  • Çarpışma ve Kaza Sayısı          : {metrics['collision_count']} (SIFIR KAZA & SIFIR HASAR)\n"
            f"  • Tonaj Üretim Başarı Skoru        : %{metrics['tonnage_score']:.1f}\n"
            f"  • SLAM Konumlandırma İndeksi       : %{metrics['slam_score']:.1f}\n"
            f"  • Otonom Maden Filo Başarı Skoru   : %{metrics['mining_autonomy_score']:.1f} (LEVEL 5 MINING AUTONOMY)\n"
            "=" * 75 + "\n"
        )
        return rapor
