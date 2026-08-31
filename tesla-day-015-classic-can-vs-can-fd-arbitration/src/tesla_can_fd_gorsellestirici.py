"""
Tesla Klasik CAN vs CAN-FD Gorsellestirici
==========================================
Bu modul, CAN 2.0B ile CAN-FD basarimini ve donanimsal arbitrasyon
surecini 6 panelli karanlik mod tani paneli olarak uretir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaCANFDGorsellestirici:
    """
    Tesla CAN-FD ve Arbitrasyon 6 panelli teshis paneli ureticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_can_fd_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA ARAÇ İÇİ İLETİŞİM PROTOKOLLERİ: KLASİK CAN VS CAN-FD]\n"
            "Modul: Gun 15 | CAN 2.0B vs CAN-FD (64-Byte Payload, 5 Mbps BRS) & Wired-AND Donanımsal Arbitrasyon",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        t_klasik = metrikler.get("klasik_sure_us", 252.0)
        t_fd = metrikler.get("can_fd_sure_us", 172.0)
        bant_klasik = metrikler.get("klasik_bant_kbps", 253.9)
        bant_fd = metrikler.get("can_fd_bant_kbps", 2976.7)
        bant_carpan = metrikler.get("bant_genisligi_carpani", 11.7)

        # 1. Panel: Çerçeve İletim Süresi (µs)
        ax1 = axes[0, 0]
        protokoller1 = ['Klasik CAN 2.0B\n(8 Byte @ 500k)', 'CAN-FD BRS\n(64 Byte @ 5M)']
        sureler1 = [t_klasik, t_fd]
        ax1.bar(protokoller1, sureler1, color=['#E06C75', '#98C379'], width=0.45)
        ax1.text(0, t_klasik + 5, f"{t_klasik:.1f} µs\n(8 Byte)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax1.text(1, t_fd + 5, f"{t_fd:.1f} µs\n(64 Byte - 8x Veri!)", ha='center', va='bottom', fontsize=9, color='#98C379', fontweight='bold')
        ax1.set_title("1. Çerçeve İletim Süresi (µs)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Süre (µs)")
        ax1.set_ylim(0, max(sureler1) * 1.35)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Efektif Veri Bant Genişliği (kbps)
        ax2 = axes[0, 1]
        ax2.bar(protokoller1, [bant_klasik, bant_fd], color=['#5c6370', '#61AFEF'], width=0.45)
        ax2.text(0, bant_klasik + 50, f"{bant_klasik:.0f} kbps", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax2.text(1, bant_fd + 50, f"{bant_fd:.0f} kbps\n({bant_carpan:.1f}x Artış)", ha='center', va='bottom', fontsize=9, color='#61AFEF', fontweight='bold')
        ax2.set_title("2. Efektif Veri Bant Genişliği (kbps)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Bant Genişliği (kbps)")
        ax2.set_ylim(0, max(bant_klasik, bant_fd) * 1.25)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Maksimum Payload Boyutu (Byte)
        ax3 = axes[0, 2]
        payload_etiket = ['Klasik CAN', 'CAN-FD']
        payload_boyut = [8, 64]
        ax3.bar(payload_etiket, payload_boyut, color=['#E5C07B', '#98C379'], width=0.45)
        ax3.text(0, 8 + 1, "8 Byte", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax3.text(1, 64 + 1, "64 Byte (8x Kapasite)", ha='center', va='bottom', fontsize=9, color='#98C379', fontweight='bold')
        ax3.set_title("3. Maksimum Payload Kapasitesi (Byte)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Byte Sayısı")
        ax3.set_ylim(0, 80)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Wired-AND Arbitrasyon Sonucu
        ax4 = axes[1, 0]
        dugumler = ['Fren Modülü\n(0x010 - ASIL-D)', 'Motor Sürücü\n(0x120 - Tork)', 'Infotainment\n(0x380 - Medya)']
        oncelik_puan = [100, 60, 20]
        renkler4 = ['#98C379', '#E5C07B', '#E06C75']
        ax4.bar(dugumler, oncelik_puan, color=renkler4, width=0.55)
        ax4.text(0, 100 + 2, "KAZANDI (0x010)\n(Kesintisiz Hat)", ha='center', va='bottom', fontsize=8, color='#98C379', fontweight='bold')
        ax4.text(1, 60 + 2, "ELENDİ (Bit 3)\n(Geri Çekildi)", ha='center', va='bottom', fontsize=8, color='#E5C07B')
        ax4.text(2, 20 + 2, "ELENDİ (Bit 1)\n(Geri Çekildi)", ha='center', va='bottom', fontsize=8, color='#E06C75')
        ax4.set_title("4. Donanımsal Arbitrasyon Öncelik Sıralaması", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_ylabel("Öncelik Seviyesi")
        ax4.set_ylim(0, 130)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: CAN-FD Çift Fazlı Hız Profili (Arbitration vs Data Phase)
        ax5 = axes[1, 1]
        fazlar = ['Arbitrasyon Fazı\n(Nominal 500 kbps)', 'Veri Fazı (BRS)\n(Yüksek Hız 5 Mbps)']
        hizlar_mbps = [0.5, 5.0]
        ax5.bar(fazlar, hizlar_mbps, color=['#61AFEF', '#98C379'], width=0.45)
        ax5.text(0, 0.5 + 0.1, "0.5 Mbps\n(Tüm Düğümler Eşzamanlı)", ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax5.text(1, 5.0 + 0.1, "5.0 Mbps (10x Hız!)\n(Yalnızca Gönderici & Alıcı)", ha='center', va='bottom', fontsize=8, color='#98C379', fontweight='bold')
        ax5.set_title("5. CAN-FD Çift Fazlı İletim Modeli (BRS)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Hız (Mbps)")
        ax5.set_ylim(0, 6.5)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: ASIL-D ve CAN-FD Kalite Özeti
        ax6 = axes[1, 2]
        metrik_etiketler = ['64-Byte Payload', '5 Mbps BRS Hız', 'Wired-AND Arb', 'Gecikmesiz İletim', 'ASIL-D']
        skorlar = [10.0, 10.0, 10.0, 9.95, 9.99]
        cubuklar6 = ax6.bar(metrik_etiketler, skorlar, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. CAN-FD Protokol Kalite Özeti", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
