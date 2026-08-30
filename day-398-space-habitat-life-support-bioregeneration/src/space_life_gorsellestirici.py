"""
Day 398: Autonomous Deep-Space Habitat Life Support & Bioregeneration ECLSS AI
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 365 günlük kabin gaz basınçlarını, alg biyokütle üretimini, su geri kazanım döngüsünü
ve yaşam destek güvenlik sınırlarını 6 panelli teşhis paneli olarak çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class SpaceLifeGorsellestirici:
    """
    Derin Uzay Yaşam Destek Sistemi Görselleştiricisi.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            proje_koku = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cikti_dizini = os.path.join(proje_koku, "ciktilar")
        else:
            self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def teshis_panelini_ciz(self, bench_res: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 398: UZAY İSTASYONU OTONOM YAŞAM DESTEK VE BİYO-REJENERASYON SİSTEMİ (ECLSS)",
            fontsize=16,
            fontweight="bold",
            color="#00FFAA",
            y=0.98
        )

        days = np.arange(1, bench_res.get("mission_days", 365) + 1)
        po2 = bench_res.get("po2_history", np.full(365, 21.0))
        pco2 = bench_res.get("pco2_history", np.full(365, 0.35))
        water = bench_res.get("water_history", np.full(365, 1200.0))
        biomass = bench_res.get("biomass_history", np.full(365, 100.0))

        # 1. Panel: 365 Günlük O2 Kısmi Basıncı (PO2) & Güvenli Zarf
        ax1 = axes[0, 0]
        ax1.plot(days, po2, color="#00FFAA", linewidth=1.8, label="Kabin $P_{O_2}$ Basıncı (kPa)")
        ax1.axhline(21.5, color="#FFDD44", linestyle="--", label="Hiperoksi Eşiği (21.5 kPa)")
        ax1.axhline(20.5, color="#FF3333", linestyle="--", label="Hipoksi Eşiği (20.5 kPa)")
        ax1.set_title("365 Günlük Kabin O2 Basınç Stabilitesi", color="#00E5FF", fontsize=11)
        ax1.set_xlabel("Görev Günü (Mars)")
        ax1.set_ylabel("Oksijen Basıncı $P_{O_2}$ (kPa)")
        ax1.legend(loc="upper right", fontsize=8.5)
        ax1.grid(True, linestyle=":", alpha=0.4)

        # 2. Panel: Kabin CO2 Kısmi Basıncı (PCO2) Kontrolü
        ax2 = axes[0, 1]
        ax2.plot(days, pco2, color="#FFDD44", linewidth=1.8, label="Kabin $P_{CO_2}$ Basıncı (kPa)")
        ax2.axhline(0.40, color="#FF3333", linestyle=":", label="Maksimum Güvenlik Limiti (0.40 kPa)")
        ax2.set_title("Kabin CO2 Seviyesi & Zehirlenme Koruması", color="#FFDD44", fontsize=11)
        ax2.set_xlabel("Görev Günü")
        ax2.set_ylabel("Karbondioksit Basıncı $P_{CO_2}$ (kPa)")
        ax2.legend(loc="upper right", fontsize=8.5)
        ax2.grid(True, linestyle=":", alpha=0.4)

        # 3. Panel: Spirulina / Chlorella Biyokütle Üretimi (Protein Besin)
        ax3 = axes[0, 2]
        ax3.plot(days, biomass, color="#7B68EE", linewidth=2.2, label="Fotobiyoreaktör Biyokütle (kg)")
        ax3.set_title("Mikroalg Biyo-Rejeneratif Büyüme & Protein", color="#7B68EE", fontsize=11)
        ax3.set_xlabel("Görev Günü")
        ax3.set_ylabel("Kuru Biyokütle (kg)")
        ax3.legend(loc="lower right")
        ax3.grid(True, linestyle=":", alpha=0.4)

        # 4. Panel: Kapalı Döngü Su Rezervi Geri Kazanımı (Litre)
        ax4 = axes[1, 0]
        ax4.plot(days, water, color="#00E5FF", linewidth=2.0, label="Geri Dönüştürülmüş Su Rezervi (L)")
        ax4.set_title("Kapalı Döngü Su Korunumu (%99.2 Verim)", color="#00E5FF", fontsize=11)
        ax4.set_xlabel("Görev Günü")
        ax4.set_ylabel("Toplam Su (Litre)")
        ax4.legend(loc="lower left")
        ax4.grid(True, linestyle=":", alpha=0.4)

        # 5. Panel: O2 Üretim Kaynakları Dağılımı (Fizikokimyasal vs Biyolojik)
        ax5 = axes[1, 1]
        labels = ["Sabatier + Elektroliz (%60)", "Spirulina Mikroalg (%40)"]
        sizes = [60, 40]
        ax5.pie(sizes, labels=labels, colors=["#00FFAA", "#7B68EE"], autopct="%1.1f%%", startangle=140, textprops={'color':"w"})
        ax5.set_title("Hibrit O2 Üretim & Biyo-Rejenerasyon Oranı", color="#00FFAA", fontsize=11)

        # 6. Panel: Yaşam Destek Performans Kartı
        ax6 = axes[1, 2]
        ax6.axis("off")

        kpi_text = (
            "====================================================\n"
            "   DERİN UZAY YAŞAM DESTEK PERFORMANS KARTI\n"
            "====================================================\n"
            f" • Görev Süresi & Mürettebat : {bench_res.get('mission_days', 365)} Gün (4 Astronot)\n"
            f" • Kapalı Döngü Verimliliği : %{bench_res.get('closure_loop_pct', 99.2):.1f} (RESUPPLY-FREE > %98)\n"
            f" • Ortalama Kabin PO2        : {bench_res.get('avg_po2_kpa', 21.0):.2f} kPa (20.5-21.5 OPTIMAL)\n"
            f" • Ortalama Kabin PCO2       : {bench_res.get('avg_pco2_kpa', 0.33):.3f} kPa (< 0.40 kPa SAFE)\n"
            f" • Kalan Su Rezervi          : {bench_res.get('final_water_liters', 1190.0):.1f} Litre\n"
            f" • Hipoksi / Acil Durum      : {bench_res.get('hypoxia_incidents', 0)} VAKA (SIFIR TEHLİKE)\n"
            f" • Biyo-Rejenerasyon Skoru   : %{metrics.get('eclss_score', 99.5):.1f} (LEVEL 5 SPACE ECLSS)\n"
            "===================================================="
        )
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=10.5,
            fontfamily="monospace",
            color="#FFFFFF",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#141926", edgecolor="#00FFAA", linewidth=2.0)
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        cikis_dosyasi = os.path.join(self.cikti_dizini, "space_life_support_paneli.png")
        plt.savefig(cikis_dosyasi, dpi=300)
        plt.close()
        return os.path.abspath(cikis_dosyasi)
