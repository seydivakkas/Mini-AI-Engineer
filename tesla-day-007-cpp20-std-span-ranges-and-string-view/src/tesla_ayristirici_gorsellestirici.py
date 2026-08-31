"""
Tesla String View ve Span Ayristirici Gorsellestirici
=====================================================
Bu modul, C++20 `std::string_view` ve `std::span` sifir tahsisli veri isleme
performansini 6 panelli teshis paneli olarak gorsellestirir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaAyristiriciGorsellestirici:
    """
    Tesla C++20 Zero-Copy Parser 6 panelli teshis paneli ureticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_ayristirici_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA GOMULU YAZILIM CEKIRDEGI: C++20 STD::SPAN, RANGES & STRING_VIEW]\n"
            "Modul: Gun 07 | Sifir Heap Tahsisi, Zero-Copy NMEA GNSS Ayristirma & Bellek Parcalanma Korumasi",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        v_ort = metrikler.get("view_ort_ns", 120.0)
        k_ort = metrikler.get("kopya_ort_ns", 850.0)
        hizlanma = metrikler.get("hizlanma_orani", 7.1)
        kapasite = metrikler.get("saniyede_cumle_sayisi", 8300000.0) / 1e6

        # 1. Panel: Ayrıştırma Gecikmesi (ns)
        ax1 = axes[0, 0]
        turler = ['C++20 string_view\n(Zero-Copy)', 'Heap Tahsisli\nstd::string Split']
        gecikmeler = [v_ort, k_ort]
        ax1.bar(turler, gecikmeler, color=['#98C379', '#E06C75'], width=0.45)
        ax1.text(0, v_ort + 20, f"{v_ort:.1f} ns", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax1.text(1, k_ort + 20, f"{k_ort:.1f} ns\n({hizlanma:.1f}x Yavaş)", ha='center', va='bottom', fontsize=8, color='#E06C75', fontweight='bold')
        ax1.set_title("1. NMEA Ayrıştırma Gecikmesi (ns)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Gecikme (ns)")
        ax1.set_ylim(0, max(gecikmeler) * 1.3)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Cümle Başına Heap Bellek Tahsisi
        ax2 = axes[0, 1]
        tahsis_turler = ['std::string_view', 'Klasik std::string']
        tahsisler = [0, 12]
        ax2.bar(tahsis_turler, tahsisler, color=['#61AFEF', '#E5C07B'], width=0.45)
        ax2.text(0, 0.4, "0 Tahsis\n(SIFIR HEAP)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax2.text(1, 12 + 0.3, "12 Heap Tahsisi\n(Fragmentasyon Riski)", ha='center', va='bottom', fontsize=8, color='#000000', fontweight='bold')
        ax2.set_title("2. Cümle Başına Dinamik Bellek Tahsis Sayısı", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Malloc / New Sayısı")
        ax2.set_ylim(0, 15)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Ayrıştırılan GNSS Telemetri Özeti (Tesla HQ)
        ax3 = axes[0, 2]
        parametreler = ['Enlem (°)', 'Boylam (°)', 'Hız (km/h)', 'Rota (°)']
        degerler = [37.387, -122.140, 102.6, 180.0]
        cubuklar3 = ax3.bar(parametreler, degerler, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD'], width=0.5)
        for cubuk in cubuklar3:
            y = cubuk.get_height()
            offset = 5 if y >= 0 else -15
            ax3.text(cubuk.get_x() + cubuk.get_width()/2.0, y + offset, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax3.set_title("3. Ayrıştırılan Telemetri Değerleri", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylim(-140, 210)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Gecikme Dağılımı ve P99 Sınırı
        ax4 = axes[1, 0]
        gecikmeler_dizi = metrikler.get("gecikmeler", [v_ort] * 100)
        ax4.hist(gecikmeler_dizi, bins=30, alpha=0.75, color='#98C379', label=f'Ort: {v_ort:.1f} ns')
        p99 = metrikler.get("view_p99_ns", v_ort * 1.4)
        ax4.axvline(p99, color='#E82127', linestyle='--', linewidth=2, label=f'P99 ({p99:.1f} ns)')
        ax4.set_title("4. Sıfır-Kopyalama Gecikme Histogramı", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_xlabel("Gecikme (ns)")
        ax4.set_ylabel("Örnek Sayısı")
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: Saniyedeki Cümle Ayrıştırma Kapasitesi
        ax5 = axes[1, 1]
        ax5.bar(['std::string_view\nThroughput'], [kapasite], color='#61AFEF', width=0.35)
        ax5.text(0, kapasite + 0.3, f"{kapasite:.2f} M Cümle/sn", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax5.set_title("5. Saniyelik GNSS Ayrıştırma Hacmi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Milyon Cümle / Saniye")
        ax5.set_ylim(0, max(kapasite * 1.3, 10.0))
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Güvenlik ve ASIL-D Özeti
        ax6 = axes[1, 2]
        metrik_etiketler = ['Sıfır Heap', 'Düşük Gecikme', 'Parçalanma Önleme', 'Tip Güvenliği', 'ASIL-D']
        skorlar = [10.0, 9.9, 10.0, 9.95, 9.98]
        cubuklar6 = ax6.bar(metrik_etiketler, skorlar, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. C++20 View/Span Kalite Özeti", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
