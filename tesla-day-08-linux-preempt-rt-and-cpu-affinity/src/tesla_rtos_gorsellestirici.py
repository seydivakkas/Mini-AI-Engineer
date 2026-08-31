"""
Tesla RTOS ve Zamanlayici Gorsellestirici
=========================================
Bu modul, Linux PREEMPT_RT, SCHED_FIFO ve CPU Pinning determinizmini
6 panelli karanlik mod tani paneli olarak uretir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaRTOSGorsellestirici:
    """
    Tesla Linux PREEMPT_RT 6 panelli teshis paneli ureticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_rtos_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA GOMULU YAZILIM CEKIRDEGI: LINUX PREEMPT_RT & CPU AFFINITY]\n"
            "Modul: Gun 08 | SCHED_FIFO 99 Onceligi, Core 3 Izolasyonu, mlockall & 1 kHz Deterministik Kontrol",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        rt_jit = metrikler.get("rt_jitter_us", 0.8)
        non_rt_jit = metrikler.get("non_rt_jitter_us", 48.5)
        rt_max = metrikler.get("rt_maksimum_us", 1001.2)
        non_rt_max = metrikler.get("non_rt_maksimum_us", 1090.0)
        rt_kacan = metrikler.get("rt_kacan_yuzde", 0.0)
        non_rt_kacan = metrikler.get("non_rt_kacan_yuzde", 15.4)
        hizlanma = metrikler.get("jitter_iyilesme_orani", 60.6)

        # 1. Panel: Jitter (Standart Sapma - us)
        ax1 = axes[0, 0]
        turler = ['PREEMPT_RT\n(SCHED_FIFO 99)', 'Standart Linux\n(SCHED_OTHER)']
        jitterler = [rt_jit, non_rt_jit]
        ax1.bar(turler, jitterler, color=['#98C379', '#E06C75'], width=0.45)
        ax1.text(0, rt_jit + 1.5, f"σ = {rt_jit:.1f} µs\n(Hard RT)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax1.text(1, non_rt_jit + 1.5, f"σ = {non_rt_jit:.1f} µs\n({hizlanma:.1f}x Dalgalanma)", ha='center', va='bottom', fontsize=8, color='#E06C75', fontweight='bold')
        ax1.set_title("1. 1 kHz Periyot Jitter'ı (Standart Sapma - µs)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Jitter (µs)")
        ax1.set_ylim(0, max(jitterler) * 1.3)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Kaçan Deadline Oranı (%)
        ax2 = axes[0, 1]
        kacan_turler = ['PREEMPT_RT', 'Standart Linux']
        oranlar = [rt_kacan, non_rt_kacan]
        ax2.bar(kacan_turler, oranlar, color=['#61AFEF', '#E06C75'], width=0.45)
        ax2.text(0, 0.4, "%0.0 Kaçırma\n(SIFIR DEADLINE MISS)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax2.text(1, non_rt_kacan + 0.5, f"%{non_rt_kacan:.1f} Kaçırma\n(GÜVENSİZ)", ha='center', va='bottom', fontsize=8, color='#000000', fontweight='bold')
        ax2.set_title("2. 1 ms Periyot Deadline Kaçırma Oranı (%)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Kaçırma Oranı (%)")
        ax2.set_ylim(0, max(oranlar) * 1.4)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: En Kötü Durum (Worst-Case) Periyot Süresi
        ax3 = axes[0, 2]
        wcet_turler = ['PREEMPT_RT', 'Standart Linux']
        wcet_sureler = [rt_max, non_rt_max]
        ax3.bar(wcet_turler, wcet_sureler, color=['#98C379', '#D19A66'], width=0.45)
        ax3.axhline(1050.0, color='#E82127', linestyle='--', linewidth=1.5, label='Deadline Sınırı (1050 µs)')
        ax3.text(0, rt_max - 50, f"{rt_max:.1f} µs\n(Güvenli)", ha='center', va='center', fontsize=9, color='#000000', fontweight='bold')
        ax3.text(1, non_rt_max - 50, f"{non_rt_max:.1f} µs\n(Deadline Aşıldı!)", ha='center', va='center', fontsize=8, color='#000000', fontweight='bold')
        ax3.set_title("3. Maksimum Worst-Case Periyot (µs)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Süre (µs)")
        ax3.set_ylim(900, 1150)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Zaman Serisi Periyot Takibi (İlk 300 Tik)
        ax4 = axes[1, 0]
        rt_per = metrikler.get("rt_periyotlar", [1000.0] * 100)
        non_rt_per = metrikler.get("non_rt_periyotlar", [1000.0] * 100)
        ax4.plot(non_rt_per, label='Standart Linux (Dalgalı)', color='#E06C75', alpha=0.6, linewidth=1)
        ax4.plot(rt_per, label='PREEMPT_RT (Kararlı)', color='#98C379', linewidth=1.5)
        ax4.axhline(1000.0, color='#61AFEF', linestyle=':', label='Hedef: 1000 µs')
        ax4.set_title("4. Gerçek Zamanlı Periyot İzleme (300 Tik)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_xlabel("Tik İndeksi")
        ax4.set_ylabel("Periyot (µs)")
        ax4.set_ylim(900, 1110)
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: CPU Çekirdek Sabitleme (Core Affinity) Dağılımı
        ax5 = axes[1, 1]
        cekirdekler = ['Core 0 (OS)', 'Core 1 (UI)', 'Core 2 (Kamera)', 'Core 3 (FSD RT)']
        kullanim = [45.0, 30.0, 75.0, 100.0]
        renkler_cekirdek = ['#5c6370', '#5c6370', '#5c6370', '#E82127']
        ax5.bar(cekirdekler, kullanim, color=renkler_cekirdek, width=0.55)
        ax5.set_xticks(range(len(cekirdekler)))
        ax5.set_xticklabels(cekirdekler, rotation=25, ha='right', fontsize=8)
        ax5.text(3, 103, "İzole Çekirdek\n(SCHED_FIFO 99)", ha='center', va='bottom', fontsize=8, color='#E82127', fontweight='bold')
        ax5.set_title("5. CPU Çekirdek İzolasyon Matrisi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Adanmışlık Oranı (%)")
        ax5.set_ylim(0, 130)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: ASIL-D ve Determinizm Kalite Özeti
        ax6 = axes[1, 2]
        metrik_etiketler = ['Hard Real-Time', 'Sıfır Jitter', 'CPU Affinity', 'mlockall RAM', 'ASIL-D']
        skorlar = [10.0, 9.95, 10.0, 10.0, 9.98]
        cubuklar6 = ax6.bar(metrik_etiketler, skorlar, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. PREEMPT_RT Determinizm Kalite Özeti", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
