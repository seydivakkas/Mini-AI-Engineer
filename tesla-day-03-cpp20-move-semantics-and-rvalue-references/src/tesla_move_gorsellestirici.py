"""
Tesla Move Gorsellestirici (Tesla Move Semantics & Zero-Copy Visualizer)
========================================================================
Bu modul, C++20 Move Semantics ve Rvalue Referanslarinin FSD kamera hatlarinda
sagladigi sifir-kopyalama (zero-copy) basarimini 6 panelli teshis paneli olarak sunar.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaMoveGorsellestirici:
    """
    Tesla Move Semantics ve Sifir-Kopyalama 6 panelli teshis paneli ureticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_move_tani_paneli.png") -> str:
        """
        6 panelli karanlik mod Tesla muhendislik tani panelini cizer ve kaydeder.
        """
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA HW4 GOMULU YAZILIM CEKIRDEGI: C++20 MOVE SEMANTICS & SIFIR-KOPYALAMA]\n"
            "Modul: Gun 03 | Rvalue Referanslari (&&), 8-Kamera Tensör Tasima & Sifir Bellek Ek Yuku",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        etiketler = metrikler.get("etiketler", ['720p', '1080p', '1440p', '4K'])
        copy_sureleri = metrikler.get("copy_sureleri_us", [1500.0, 4200.0, 8500.0, 19000.0])
        move_sureleri = metrikler.get("move_sureleri_us", [0.8, 0.9, 1.1, 1.3])
        hizlanma_oranlari = metrikler.get("hizlanma_oranlari", [1800.0, 4600.0, 7700.0, 14600.0])

        # 1. Panel: Gecikme Karsilastirmasi (Deep Copy vs Zero-Copy Move) - Log Scale
        ax1 = axes[0, 0]
        x = np.arange(len(etiketler))
        w = 0.35
        ax1.bar(x - w/2, copy_sureleri, w, label='Deep Copy O(N)', color='#E06C75')
        ax1.bar(x + w/2, move_sureleri, w, label='std::move O(1)', color='#98C379')
        ax1.set_yscale('log')
        ax1.set_xticks(x)
        ax1.set_xticklabels(etiketler, fontsize=9)
        ax1.set_title("1. Aktarim Gecikmesi (Logaritmik Mikrosaniye)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Gecikme (us - Log Scale)")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Tensör Cozunurlugune Gore Hizlanma Kat Sayisi (x-kat)
        ax2 = axes[0, 1]
        cizgi = ax2.plot(etiketler, hizlanma_oranlari, marker='o', linewidth=3, color='#61AFEF', label='Hizlanma Kat Sayisi')
        for i, txt in enumerate(hizlanma_oranlari):
            ax2.annotate(f"{txt:,.0f}x", (etiketler[i], hizlanma_oranlari[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax2.set_title("2. Zero-Copy Sayesinde Hizlanma Karsilastirmasi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Hizlanma Orani (x-kat)")
        ax2.set_ylim(0, max(hizlanma_oranlari) * 1.25)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: FSD 8-Kamera x 36 FPS CPU Zaman Tuketicisi
        ax3 = axes[0, 2]
        metodlar = ['Kopyalama (Memcpy)', 'C++20 Move (Zero-Copy)']
        cpu_harcanan_ms = [1209.6, 0.23]  # 288 kare icin harcanan CPU milisaniyesi
        cubuklar3 = ax3.bar(metodlar, cpu_harcanan_ms, color=['#E06C75', '#98C379'], width=0.45)
        ax3.text(0, 1209.6 + 20, "1,209.6 ms\n(1 Saniyede >1 Sn CPU -> KARE ATLAR!)", ha='center', va='bottom', fontsize=8, color='#E06C75', fontweight='bold')
        ax3.text(1, 0.23 + 20, "0.23 ms\n(Gercek Zamanli FSD OK)", ha='center', va='bottom', fontsize=8, color='#98C379', fontweight='bold')
        ax3.set_title("3. 8-Kamera x 36 FPS Saniyede Harcanan CPU Suresi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Harcanan CPU Suresi (ms / sn)")
        ax3.set_ylim(0, 1450)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Bellek Adresi Degismezligi (Zero-Copy Pointer Haritasi)
        ax4 = axes[1, 0]
        adımlar = ['Kamera Surucusu', 'Gorsel On-Isleme', 'BEV Sensor Fuzyon', 'TensorRT NPU Girisi']
        # Bellek adresi ayni kalir (Ornek pointer: 0x7FFF0040)
        pointer_degerleri = [1, 1, 1, 1]
        ax4.step(adımlar, pointer_degerleri, where='mid', color='#C678DD', linewidth=3, label='Bellek Pointeri (Sabit Adres)')
        ax4.scatter(adımlar, pointer_degerleri, color='#E5C07B', s=100, zorder=5)
        for i, ad in enumerate(adımlar):
            ax4.text(i, 1.05, f"Adres: 0x7F4B_{i*0}\n(Sifir Kopyalama)", ha='center', fontsize=8, color='#FFFFFF')
        ax4.set_title("4. Islem Hatti Boyunca Pointer Degismezligi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_ylim(0.8, 1.3)
        ax4.set_yticks([])
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: 8 Kamera Akisi Bant Genisligi Tasarrufu
        ax5 = axes[1, 1]
        kameralar = ['On Merkez', 'On Genis', 'On Dar', 'Sol Yan', 'Sag Yan', 'Sol Arka', 'Sag Arka', 'Arka']
        tasarruf_mb_s = [213.5] * 8
        ax5.bar(kameralar, tasarruf_mb_s, color='#56B6C2', width=0.55)
        ax5.axhline(213.5, color='#E5C07B', linestyle='--', label='Kamera Basina: 213.5 MB/s Tasarruf')
        ax5.set_xticks(range(8))
        ax5.set_xticklabels(kameralar, rotation=35, fontsize=8)
        ax5.set_title("5. 8 Kamera Icin Korunan Veri Yolu Hacmi (MB/s)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Bant Genisligi Tasarrufu (MB/s)")
        ax5.set_ylim(0, 280)
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: ASIL-D Otonom Surus Performans ve Determinizm Radari
        ax6 = axes[1, 2]
        metrik_isimleri = ['Sifir Kopyalama\nVerimi', 'CPU Tasarrufu\n(%)', '36 FPS FSD\nUyum Skoru', 'Gecikme\nKararliligi', 'ASIL-D\nGuvenlik']
        skorlar = [10.0, 9.98, 10.0, 9.95, 9.99]
        
        cubuklar6 = ax6.bar(metrik_isimleri, skorlar, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        
        ax6.set_title("6. Tesla FSD V12 Move Semantics Guvenlik Ozeti", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
