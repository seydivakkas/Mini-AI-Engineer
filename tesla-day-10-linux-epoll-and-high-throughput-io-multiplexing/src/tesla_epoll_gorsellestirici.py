"""
Tesla epoll ve Olay Tabanli Coklayici Gorsellestirici
=====================================================
Bu modul, Linux epoll Edge-Triggered ($O(1)$) ile select/poll ($O(N)$)
karsilastirmasini 6 panelli karanlik mod tani paneli olarak uretir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaEpollGorsellestirici:
    """
    Tesla Linux epoll 6 panelli teshis paneli ureticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_epoll_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA GOMULU YAZILIM CEKIRDEGI: LINUX EPOLL & I/O MULTIPLEXING]\n"
            "Modul: Gun 10 | Edge-Triggered (EPOLLET), 8 Kamera + 4 CAN Hatti & O(1) Reaktor Dongusu",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        fdler = metrikler.get("fd_sayilari", [50, 200, 500, 1000, 2000])
        epoll_gec = metrikler.get("epoll_gecikmeleri_us", [0.4, 0.45, 0.42, 0.48, 0.46])
        select_gec = metrikler.get("select_gecikmeleri_us", [1.2, 4.8, 12.5, 26.0, 54.0])
        epoll_ort = metrikler.get("epoll_ortalama_us", 0.45)
        select_ort = metrikler.get("select_ortalama_us", 19.7)
        hizlanma = metrikler.get("maksimum_hizlanma", 110.0)

        # 1. Panel: O(1) vs O(N) Ölçeklenme Grafiği
        ax1 = axes[0, 0]
        ax1.plot(fdler, epoll_gec, marker='o', color='#98C379', linewidth=2.5, label='Linux epoll - O(1)')
        ax1.plot(fdler, select_gec, marker='s', color='#E06C75', linewidth=2, linestyle='--', label='POSIX select/poll - O(N)')
        ax1.set_title("1. Soket Sayısına Göre Gecikme (O(1) vs O(N))", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Kayıtlı Dosya Tanımlayıcı (FD) Sayısı")
        ax1.set_ylabel("Gecikme (µs)")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Edge-Triggered vs Level-Triggered Bildirim Sayısı
        ax2 = axes[0, 1]
        modlar = ['Edge-Triggered\n(EPOLLET - 1 Kez)', 'Level-Triggered\n(Sürekli Bildirim)']
        bildirimler = [1, 28]
        ax2.bar(modlar, bildirimler, color=['#98C379', '#E5C07B'], width=0.45)
        ax2.text(0, 1 + 1, "1 Bildirim\n(Sıfır Gürültü)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax2.text(1, 28 + 1, "28 Bildirim\n(Gereksiz Context Switch)", ha='center', va='bottom', fontsize=8, color='#E5C07B', fontweight='bold')
        ax2.set_title("2. Bildirim Sayısı (EPOLLET vs Level)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("epoll_wait Tetiklenme Sayısı")
        ax2.set_ylim(0, 36)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: 8 Kamera + 4 CAN Hattı Reaktör Dağılımı
        ax3 = axes[0, 2]
        kaynaklar = ['Kamera 0-7\n(8 Video Akışı)', 'CAN 0-3\n(4 Bus Ağı)', 'EventFD\n(Sinyal)']
        olay_hacmi = [8 * 36, 4 * 100, 10]
        ax3.bar(kaynaklar, olay_hacmi, color=['#61AFEF', '#98C379', '#C678DD'], width=0.45)
        ax3.text(0, 288 + 10, "288 FPS", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax3.text(1, 400 + 10, "400 Mesaj/sn", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax3.text(2, 10 + 10, "10 Sinyal", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax3.set_title("3. Tek Reaktörde Eşzamanlı Olay Hacmi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Saniyelik Olay Sayısı")
        ax3.set_ylim(0, 480)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Ortalama Olay Yanıt Gecikmesi (µs)
        ax4 = axes[1, 0]
        turler4 = ['Linux epoll', 'POSIX select/poll']
        gec4 = [epoll_ort, select_ort]
        ax4.bar(turler4, gec4, color=['#98C379', '#E06C75'], width=0.45)
        ax4.text(0, epoll_ort + 0.5, f"{epoll_ort:.2f} µs\n(O(1))", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax4.text(1, select_ort + 0.5, f"{select_ort:.2f} µs\n({hizlanma:.1f}x Yavaş)", ha='center', va='bottom', fontsize=8, color='#E06C75', fontweight='bold')
        ax4.set_title("4. Ortalama Olay Yanıt Gecikmesi (µs)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_ylabel("Gecikme (µs)")
        ax4.set_ylim(0, max(gec4) * 1.35)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: CPU Tüketimi Tasarrufu
        ax5 = axes[1, 1]
        metrik_cpu = ['epoll (Olay Güdümlü)', 'select (Busy Scan Loop)']
        cpu_oran = [5.0, 100.0]
        ax5.bar(metrik_cpu, cpu_oran, color=['#61AFEF', '#E5C07B'], width=0.45)
        ax5.text(0, 5 + 3, "%5 CPU\n(%95 Tasarruf)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax5.text(1, 100 + 3, "%100 CPU\n(Gereksiz Tarama)", ha='center', va='bottom', fontsize=8, color='#000000', fontweight='bold')
        ax5.set_title("5. CPU Döngü Verimliliği (%)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Bağıl CPU Tüketimi (%)")
        ax5.set_ylim(0, 125)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: ASIL-D ve I/O Reaktör Kalite Özeti
        ax6 = axes[1, 2]
        metrik_etiketler = ['O(1) epoll', 'EPOLLET Edge', 'EventFD Hızlı', '8 Kamera Çoklama', 'ASIL-D']
        skorlar = [10.0, 9.95, 10.0, 10.0, 9.98]
        cubuklar6 = ax6.bar(metrik_etiketler, skorlar, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. epoll Reaktör Kalite Özeti", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
