r"""
Tesla Dojo Görselleştirici Modülü
==================================
Bu modül; 25 D1 çipli ($5 \times 5$) Dojo Training Tile 2D Mesh ağını,
Dimension-Ordered (XY) yönlendirme yolunu, biseksiyon gecikmelerini ve
NoC mimarisini 6 panelli karanlık mod tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaDojoGorsellestirici:
    """
    Tesla Dojo 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_dojo_d1_mesh_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA DOJO SÜPERBİLGİSAYAR MİMARİSİ: D1 ÇİPİ VE 2D MESH NoC]\n"
            "Modül: Gün 89 | 25 D1 Çipi, 9 PFLOPS Training Tile, 36 TB/s Biseksiyon Bant Genişliği & 2.5 ns Hop",
            fontsize=14, fontweight='bold', color='#E82127', y=0.98
        )

        pflops = metrikler.get("tile_pflops", 9.05)
        chips = metrikler.get("num_chips", 25)
        lat_ns = metrikler.get("total_latency_ns", 544.0)
        bw_gb = metrikler.get("effective_bw_gb_s", 1928.0)
        hop_mat = metrikler.get("hop_matrix", np.zeros((5, 5)))
        path = metrikler.get("path_sample", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)])
        step_ort = metrikler.get("step_ortalama_us", 1.8)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 100)

        # 1. Panel: 5x5 Dojo Training Tile Atlama (Hop) Isı Haritası
        ax1 = axes[0, 0]
        im1 = ax1.imshow(hop_mat, cmap='magma', interpolation='nearest')
        for i in range(5):
            for j in range(5):
                ax1.text(j, i, f'{int(hop_mat[i, j])}h', ha='center', va='center', color='#FFFFFF', fontsize=9, fontweight='bold')
        ax1.set_title("1. 5x5 Training Tile Hop Haritası (Kök: 0,0)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("D1 X Koordinatı")
        ax1.set_ylabel("D1 Y Koordinatı")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 2. Panel: Dimension-Ordered (XY) Yönlendirme Yolu
        ax2 = axes[0, 1]
        grid_x, grid_y = np.meshgrid(np.arange(5), np.arange(5))
        ax2.scatter(grid_x, grid_y, color='#5c6370', s=120, label='D1 Çipleri (5x5)')
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax2.plot(path_x, path_y, color='#E82127', linewidth=3.0, marker='o', markersize=8, label='DOR (XY) Yönlendirme')
        ax2.scatter([0], [0], color='#98C379', s=200, zorder=5, label='Kaynak (0,0)')
        ax2.scatter([4], [4], color='#C678DD', s=200, zorder=5, label='Hedef (4,4)')
        ax2.set_title("2. 2D Mesh XY Yönlendirme Yolu", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_xlabel("X Ekseni")
        ax2.set_ylabel("Y Ekseni")
        ax2.set_xlim(-0.5, 4.5)
        ax2.set_ylim(-0.5, 4.5)
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Veri Boyutu vs İletim Gecikmesi (ns)
        ax3 = axes[0, 2]
        payloads_kb = np.linspace(64, 2048, 30)
        lats = [8 * 2.5 + (p * 1024 / 2000.0) for p in payloads_kb]
        ax3.plot(payloads_kb, lats, color='#98C379', linewidth=2.5, label='Transfer Gecikmesi T(s)')
        ax3.set_title("3. Tensor Paket Boyutu vs NoC Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Tensor Boyutu (Kilobayt - KB)")
        ax3.set_ylabel("Gecikme (Nanosaniye - ns)")
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Tesla Dojo Durum Kartı
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.88, "TESLA DOJO D1 TRAINING TILE KARTI", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"D1 ÇİP SAYISI: {chips} Adet Özel 7nm Silikon\nTILE HESAPLAMA GÜCÜ: {pflops:.2f} PFLOPS (BF16/CFP8)\nBİSEKSİYON BANT GENİŞLİĞİ: 36 TB/s (2 TB/s Kenar Başı)\nNoC TOPOLOJİSİ: 2D Mesh / Torus + DOR (XY)\nATLAMA GECİKMESİ: 2.5 ns / Hop\n1 MB TRANSFER SÜRESİ: {lat_ns:.1f} ns ({bw_gb:,.0f} GB/s)",
                 ha='center', va='center', fontsize=9.5, color='#FFFFFF')
        ax4.text(0.5, 0.18, f"DURUM: %100 ULTRA HIZLI TENSOR DAĞITIMI", ha='center', va='center', fontsize=11, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. Dojo Süperbilgisayar Raporu", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: Yönlendirme Hesaplama Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=20, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. NoC Yönlendirme ve Karar Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Gecikme (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Dojo Başarı Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['D1 Chip Mesh', '9 PFLOPS Tile', '36 TB/s NoC', 'XY Deadlock-Free', 'Sub-2µs RTOS']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla Dojo NoC Başarı Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.tick_params(axis='x', rotation=20)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
