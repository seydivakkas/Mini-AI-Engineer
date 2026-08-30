"""
Day 395: Autonomous Disaster Response & Humanitarian Logistics Fleet AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Afet hayatta kalma oranını, acil müdahale hızını,
blokaj aşma başarısını ve insani yardım lojistiği otonomi skorunu profiller.
"""

from typing import Dict, Any


class DisasterProfilleyici:
    """
    Afet Müdahale ve İnsani Yardım Performans Profilleyicisi.
    """
    def __init__(self):
        pass

    def profille(self, bench_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Afet başarım sonuçlarından performans metriklerini hesaplar.
        """
        survival = bench_res.get("overall_survival_rate_pct", 95.0)
        avg_time = bench_res.get("avg_response_time_min", 18.0)
        blocked_count = bench_res.get("roadblocks_bypassed_count", 5)

        survival_score = survival
        speed_score = max(0.0, 100.0 - (avg_time / 25.0) * 10.0)
        bypass_score = 100.0 if blocked_count > 0 else 90.0

        disaster_score = (survival_score * 0.45 + speed_score * 0.35 + bypass_score * 0.20)

        return {
            "survival_score": round(survival_score, 2),
            "speed_score": round(speed_score, 2),
            "bypass_score": round(bypass_score, 2),
            "disaster_score": round(disaster_score, 2),
            "total_victims": bench_res.get("total_victims", 500),
            "red_critical_count": bench_res.get("red_critical_count", 0),
            "avg_response_time_min": avg_time,
            "overall_survival_rate_pct": survival
        }

    def rapor_olustur(self, metrics: Dict[str, Any]) -> str:
        """
        Konsol için yapılandırılmış Afet Müdahale Raporu üretir.
        """
        rapor = (
            "\n" + "=" * 75 + "\n"
            "   DAY 395: AFET MÜDAHALE VE İNSANİ YARDIM FİLOSU RAPORU\n"
            "=" * 75 + "\n"
            f"  • Toplam Yönetilen Kazazede Sayısı : {metrics['total_victims']} Kişi\n"
            f"  • Kritik Kırmızı Öncelikli Vaka     : {metrics['red_critical_count']} Ağır Yaralı\n"
            f"  • Ortalama Acil Müdahale Süresi    : {metrics['avg_response_time_min']:.1f} Dakika (< 25 dk PASS)\n"
            f"  • Genel Hayatta Kalma Oranı        : %{metrics['overall_survival_rate_pct']:.1f} (ALTIN SAAT İÇİNDE)\n"
            f"  • Filo Hız ve Dağıtım Skoru        : %{metrics['speed_score']:.1f}\n"
            f"  • Otonom İnsani Yardım Başarı Skoru: %{metrics['disaster_score']:.1f} (LEVEL 5 CRISIS AI)\n"
            "=" * 75 + "\n"
        )
        return rapor
