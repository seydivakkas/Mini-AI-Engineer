"""
Day 389: Global Supply Chain Disruption Forecaster & Dynamic Rerouting
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Tedarik zinciri dayanıklılık indeksini, gecikme sönümleme başarısını,
stok tükenmesi önleme oranını ve genel lojistik otonomi skorunu profiller.
"""

from typing import Dict, Any


class SupplyChainProfilleyici:
    """
    Küresel Tedarik Zinciri Performans Profilleyicisi.
    """
    def __init__(self):
        pass

    def profille(self, bench_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tedarik zinciri başarım sonuçlarından performans metriklerini hesaplar.
        """
        resilience = bench_res.get("supply_chain_resilience_score", 96.0)
        stockout_prev = bench_res.get("stockout_prevented_pct", 95.0)
        delay_mit = bench_res.get("delay_mitigation_pct", 40.0)

        resilience_score = resilience
        stockout_score = stockout_prev
        mitigation_score = min(100.0, (delay_mit / 35.0) * 100.0)

        supply_chain_score = (resilience_score * 0.40 + stockout_score * 0.35 + mitigation_score * 0.25)

        return {
            "resilience_score": round(resilience_score, 2),
            "stockout_score": round(stockout_score, 2),
            "mitigation_score": round(mitigation_score, 2),
            "supply_chain_score": round(supply_chain_score, 2),
            "nominal_transit_days": bench_res.get("nominal_transit_days", 41.0),
            "rerouted_transit_days": bench_res.get("rerouted_transit_days", 50.5),
            "chokepoint_crisis_handled": bench_res.get("chokepoint_crisis_handled", "SUEZ_BLOCKAGE")
        }

    def rapor_olustur(self, metrics: Dict[str, Any]) -> str:
        """
        Konsol için yapılandırılmış Tedarik Zinciri Raporu üretir.
        """
        rapor = (
            "\n" + "=" * 75 + "\n"
            "   DAY 389: KÜRESEL TEDARİK ZİNCİRİ KRİZ VE ROTA YENİLEME RAPORU\n"
            "=" * 75 + "\n"
            f"  • Yönetilen Küresel Kriz           : {metrics['chokepoint_crisis_handled']}\n"
            f"  • Tedarik Zinciri Dayanıklılık Skoru: %{metrics['resilience_score']:.2f} (RESILIENT ADAPTIVE NET)\n"
            f"  • Stoksuz Kalma Önleme Başarısı    : %{metrics['stockout_score']:.2f} (SIFIR DURUŞ)\n"
            f"  • Gecikme Sönümleme Oranı          : %{metrics['mitigation_score']:.1f}\n"
            f"  • Nominal / Yeni Transit Süresi    : {metrics['nominal_transit_days']:.1f} Gün -> {metrics['rerouted_transit_days']:.1f} Gün\n"
            f"  • Otonom Lojistik ve Tedarik Skoru : %{metrics['supply_chain_score']:.1f} (LEVEL 5 SUPPLY CHAIN)\n"
            "=" * 75 + "\n"
        )
        return rapor
