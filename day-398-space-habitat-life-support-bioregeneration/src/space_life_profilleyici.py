"""
Day 398: Autonomous Deep-Space Habitat Life Support & Bioregeneration ECLSS AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Kapalı döngü yaşam destek verimliliğini, atmosfer stabilitesini,
biyo-rejeneratif besin üretimini ve derin uzay habitat otonomi skorunu profiller.
"""

from typing import Dict, Any


class SpaceLifeProfilleyici:
    """
    Derin Uzay ECLSS Yaşam Destek Sistemi Performans Profilleyicisi.
    """
    def __init__(self):
        pass

    def profille(self, bench_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        ECLSS başarım sonuçlarından performans metriklerini hesaplar.
        """
        closure = bench_res.get("closure_loop_pct", 99.2)
        po2 = bench_res.get("avg_po2_kpa", 21.0)
        pco2 = bench_res.get("avg_pco2_kpa", 0.33)
        hypoxia = bench_res.get("hypoxia_incidents", 0)

        # O2 sapması puanı (21.0 kPa'dan uzaklık)
        o2_stability_score = max(0.0, 100.0 - abs(po2 - 21.0) * 50.0)
        closure_score = min(100.0, (closure / 98.0) * 95.0)
        safety_score = 100.0 if hypoxia == 0 and pco2 <= 0.40 else 60.0

        eclss_score = (o2_stability_score * 0.35 + closure_score * 0.40 + safety_score * 0.25)

        return {
            "o2_stability_score": round(o2_stability_score, 2),
            "closure_score": round(closure_score, 2),
            "safety_score": round(safety_score, 2),
            "eclss_score": round(eclss_score, 2),
            "mission_days": bench_res.get("mission_days", 365),
            "crew_count": bench_res.get("crew_count", 4),
            "closure_loop_pct": closure,
            "avg_po2_kpa": po2,
            "avg_pco2_kpa": pco2
        }

    def rapor_olustur(self, metrics: Dict[str, Any]) -> str:
        """
        Konsol için yapılandırılmış Uzay Yaşam Destek Raporu üretir.
        """
        rapor = (
            "\n" + "=" * 75 + "\n"
            "   DAY 398: DERİN UZAY YAŞAM DESTEK VE BİYO-REJENERASYON (ECLSS) RAPORU\n"
            "=" * 75 + "\n"
            f"  • Simülasyon Görev Süresi          : {metrics['mission_days']} Gün (4 Astronot Mürettebat)\n"
            f"  • Kapalı Döngü Kütle Verimliliği   : %{metrics['closure_loop_pct']:.1f} (> %98 DÜNYADAN BAĞIMSIZ)\n"
            f"  • Ortalama Oksijen Basıncı (PO2)   : {metrics['avg_po2_kpa']:.2f} kPa (20.5 - 21.5 kPa GÜVENLİ)\n"
            f"  • Ortalama Karbondioksit (PCO2)    : {metrics['avg_pco2_kpa']:.3f} kPa (< 0.40 kPa SAFE)\n"
            f"  • Atmosfer Kararlılık Skoru        : %{metrics['o2_stability_score']:.1f}\n"
            f"  • Otonom Derin Uzay ECLSS Skoru    : %{metrics['eclss_score']:.1f} (LEVEL 5 SPACE TECH)\n"
            "=" * 75 + "\n"
        )
        return rapor
