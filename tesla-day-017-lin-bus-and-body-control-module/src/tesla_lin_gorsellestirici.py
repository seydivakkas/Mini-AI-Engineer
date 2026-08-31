"""
Tesla LIN ve BCM Gorsellestirici
================================
Bu modul, LIN 2.x cerceve iletimini, Master cizelgeleme tablosunu
ve BCM govde kontrol durumlarini 6 panelli karanlik mod tani paneli olarak uretir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaLINGorsellestirici:
    """
    Tesla LIN Veri Yolu ve BCM 6 panelli teshis paneli ureticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_lin_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA ARAÇ İÇİ İLETİŞİM PROTOKOLLERİ: LIN VERİ YOLU & GÖVDE KONTROLÜ (BCM)]\n"
            "Modul: Gun 17 | LIN 2.x Master-Slave, Schedule Table, PID Parite (P0, P1) & Tek Hat (Single-Wire) 12V BCM",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        t_19k2 = metrikler.get("lin_19k2_sure_ms", 3.28)
        t_9k6 = metrikler.get("lin_9k6_sure_ms", 6.56)
        pid_ort = metrikler.get("pid_ortalama_us", 0.45)

        # 1. Panel: LIN Baudrate İletim Süresi (ms)
        ax1 = axes[0, 0]
        baud_etiket = ['LIN 19.2 kbps\n(Standart Hız)', 'LIN 9.6 kbps\n(Düşük Hız)']
        baud_sure = [t_19k2, t_9k6]
        ax1.bar(baud_etiket, baud_sure, color=['#98C379', '#E5C07B'], width=0.45)
        ax1.text(0, t_19k2 + 0.2, f"{t_19k2:.2f} ms", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax1.text(1, t_9k6 + 0.2, f"{t_9k6:.2f} ms\n(2x Yavaş)", ha='center', va='bottom', fontsize=9, color='#E5C07B')
        ax1.set_title("1. LIN Çerçeve İletim Süresi (ms)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Süre (ms)")
        ax1.set_ylim(0, max(baud_sure) * 1.35)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Master Çizelgeleme Tablosu (Schedule Table Zaman Aralıkları)
        ax2 = axes[0, 1]
        gorevler = ['Pencere (0x32)', 'Silecek (0x0A)', 'Ambiyans (0x28)', 'Koltuk (0x14)']
        araliklar = [10.0, 20.0, 50.0, 100.0]
        ax2.bar(gorevler, araliklar, color=['#61AFEF', '#98C379', '#C678DD', '#E5C07B'], width=0.55)
        for i, val in enumerate(araliklar):
            ax2.text(i, val + 2, f"{val:.0f} ms", ha='center', va='bottom', fontsize=8, color='#FFFFFF', fontweight='bold')
        ax2.set_title("2. Master Schedule Table Görev Aralıkları", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Periyot (ms)")
        ax2.set_ylim(0, 120)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: PID Parite Doğrulama ve Güvenlik
        ax3 = axes[0, 2]
        pid_senaryo = ['Geçerli PID (P0,P1)', 'Bozuk P0 Parite', 'Bozuk P1 Parite']
        pid_kabul = [100.0, 0.0, 0.0]
        ax3.bar(pid_senaryo, pid_kabul, color=['#98C379', '#E82127', '#E82127'], width=0.5)
        ax3.text(0, 102, "%100 Kabul", ha='center', va='bottom', fontsize=8, color='#98C379', fontweight='bold')
        ax3.text(1, 4, "%100 Reddedildi", ha='center', va='bottom', fontsize=8, color='#E82127', fontweight='bold')
        ax3.text(2, 4, "%100 Reddedildi", ha='center', va='bottom', fontsize=8, color='#E82127', fontweight='bold')
        ax3.set_title("3. Protected ID (PID) Parite Denetimi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Kabul Oranı (%)")
        ax3.set_ylim(0, 125)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: BCM Canlı Aktüatör Durumları
        ax4 = axes[1, 0]
        bcm_aygitlar = ['Pencere\n(%80 Açık)', 'Koltuk\n(200 mm)', 'Silecek\n(Kademe 2)', 'Ambiyans\n(Kırmızı)']
        bcm_degerler = [80.0, 200.0 / 3.0, 2.0 * 25.0, 100.0]
        ax4.bar(bcm_aygitlar, bcm_degerler, color=['#61AFEF', '#E5C07B', '#98C379', '#E82127'], width=0.5)
        ax4.set_title("4. BCM Gövde Aktüatör Durumları", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_ylabel("Normalize Değer")
        ax4.set_ylim(0, 120)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: Kablolama ve Maliyet Kazancı (CAN vs LIN)
        ax5 = axes[1, 1]
        karsilastirma = ['CAN Veri Yolu\n(2 Hat + Transceiver)', 'LIN Veri Yolu\n(Tek Hat 12V + UART)']
        maliyet_skoru = [100.0, 28.0]
        ax5.bar(karsilastirma, maliyet_skoru, color=['#E06C75', '#98C379'], width=0.45)
        ax5.text(0, 100 + 3, "100% (Referans)", ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax5.text(1, 28 + 3, "%28 Maliyet\n(%72 Tasarruf!)", ha='center', va='bottom', fontsize=8, color='#98C379', fontweight='bold')
        ax5.set_title("5. Donanım & Kablo Maliyet Tasarrufu", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Göreceli Maliyet (%)")
        ax5.set_ylim(0, 130)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: ASIL ve LIN Kalite Özeti
        ax6 = axes[1, 2]
        metrik_etiketler = ['Master Schedule', 'PID Parite P0/P1', 'Enhanced Csum', 'BCM Sürücüleri', 'Maliyet Verimi']
        skorlar = [10.0, 10.0, 10.0, 10.0, 10.0]
        cubuklar6 = ax6.bar(metrik_etiketler, skorlar, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#56B6C2'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. LIN ve BCM Kalite Özeti", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
