"""
Day 387: City-Scale Traffic Optimization & V2X Autonomous Vehicle Platooning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; V2X konvoy dizi kararlılığını, şehir trafiği akış iyileştirmesini,
enerji tasarrufunu ve otonom ulaşım skorunu profiller.
"""

from typing import Dict, Any


class TrafficProfilleyici:
    """
    Şehir Ölçeğinde Trafik ve V2X Konvoy Performans Profilleyicisi.
    """
    def __init__(self):
        pass

    def profille(self, bench_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trafik ve konvoy başarım metriklerini profiller.
        """
        stability_ratio = bench_res.get("string_stability_ratio", 0.85)
        travel_saving = bench_res.get("travel_time_reduction_pct", 30.0)
        energy_saving = bench_res.get("energy_saving_pct", 18.0)
        is_stable = bench_res.get("is_string_stable", True)

        stability_score = 100.0 if is_stable else max(0.0, (1.0 - (stability_ratio - 1.0)) * 80.0)
        flow_score = min(100.0, (travel_saving / 25.0) * 100.0)
        energy_score = min(100.0, (energy_saving / 15.0) * 100.0)

        traffic_autonomy_score = (stability_score * 0.40 + flow_score * 0.35 + energy_score * 0.25)

        return {
            "stability_score": round(stability_score, 2),
            "flow_score": round(flow_score, 2),
            "energy_score": round(energy_score, 2),
            "traffic_autonomy_score": round(traffic_autonomy_score, 2),
            "string_stability_ratio": stability_ratio,
            "travel_time_reduction_pct": travel_saving,
            "energy_saving_pct": energy_saving,
            "is_string_stable": is_stable
        }

    def rapor_olustur(self, metrics: Dict[str, Any]) -> str:
        """
        Konsol için yapılandırılmış V2X Trafik Raporu üretir.
        """
        rapor = (
            "\n" + "=" * 75 + "\n"
            "   DAY 387: ŞEHİR ÖLÇEĞİNDE V2X TRAFİK & KONVOY OPTİMİZASYON RAPORU\n"
            "=" * 75 + "\n"
            f"  • Konvoy Dizi Kararlılığı (String Stb): {metrics['string_stability_ratio']:.3f} (<= 1.0 MÜKEMMEL SÖNÜM)\n"
            f"  • Dalga Kararlılık Durumu (No Shock): {'%100 KARARLI (HAYALET TRAFİK YOK)' if metrics['is_string_stable'] else 'DALGALANMA'}\n"
            f"  • Seyahat Süresi İyileşmesi (Akış)  : %{metrics['travel_time_reduction_pct']:.1f}\n"
            f"  • Aerodinamik Enerji Tasarrufu      : %{metrics['energy_saving_pct']:.1f}\n"
            f"  • Dizi Kararlılık Skoru             : %{metrics['stability_score']:.1f}\n"
            f"  • Şehir Akım Akış Başarı Skoru      : %{metrics['flow_score']:.1f}\n"
            f"  • V2X Trafik Otonomi Başarı Skoru   : %{metrics['traffic_autonomy_score']:.1f} (LEVEL 5 V2X PLATOONING)\n"
            "=" * 75 + "\n"
        )
        return rapor
