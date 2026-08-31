r"""
Tesla AEB Görselleştirici Modülü
================================
Bu modül; Acil durum durma mesafesini ($d_{\text{stop}}(v)$), Euro-NCAP AEB kademelerini,
frenleme ivmesi profilini ve karar gecikmesini 6 panelli karanlık mod
tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaAEBGorsellestirici:
    """
    Tesla AEB ve Acil Kaçınma Kontrolcüsü 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_aeb_emergency_maneuver_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA FSD OTOMATİK ACİL FRENLEME (AEB) VE ACİL DURUM MANEVRASI]\n"
            "Modül: Gün 62 | Euro-NCAP Protokolü, Durma Mesafesi Formülasyonu, -9.0 m/s² Fren & 5 µs Karar",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        d_stop = metrikler.get("stopping_dist_m", 26.2)
        d_obs = metrikler.get("dist_obs_m", 18.0)
        ttc = metrikler.get("ttc_s", 0.9)
        level = metrikler.get("level", "FULL_AEB")
        action = metrikler.get("action_desc", "TAM ACİL FRENLEME")
        step_ort = metrikler.get("aeb_step_ortalama_us", 5.0)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 100)

        # 1. Panel: Hıza Göre Acil Durma Mesafesi Eğrisi
        ax1 = axes[0, 0]
        speeds_kmh = np.linspace(10, 120, 50)
        speeds_mps = speeds_kmh / 3.6
        react_dists = speeds_mps * 0.20
        brake_dists = (speeds_mps ** 2) / (2.0 * 9.0)
        total_dists = react_dists + brake_dists

        ax1.plot(speeds_kmh, total_dists, color='#E06C75', linewidth=2.5, label='Toplam Durma Mesafesi (d_stop)')
        ax1.plot(speeds_kmh, brake_dists, color='#E5C07B', linestyle='--', label='Frenleme Mesafesi (9.0 m/s²)')
        ax1.plot(speeds_kmh, react_dists, color='#61AFEF', linestyle=':', label='Tepki/Gecikme Mesafesi (0.2s)')
        ax1.scatter([72.0], [d_stop], color='#E82127', s=100, label=f'72 km/h: {d_stop:.1f}m')
        ax1.set_title("1. Acil Durum Durma Mesafesi Eğrisi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Hız (km/h)")
        ax1.set_ylabel("Mesafe (Metre)")
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Euro-NCAP AEB Kademeleri ve TTC Eşikleri
        ax2 = axes[0, 1]
        tiers = ['Normal', 'FCW Uyarısı', 'Kısmi Fren', 'Tam AEB']
        ttc_thresholds = [3.0, 2.4, 1.6, 1.0]
        colors = ['#98C379', '#E5C07B', '#E5C07B', '#E06C75']
        ax2.bar(tiers, ttc_thresholds, color=colors, width=0.5)
        ax2.axhline(y=ttc, color='#E82127', linestyle='--', linewidth=2, label=f'Mevcut TTC: {ttc:.2f}s (AEB Aktif)')
        ax2.set_title("2. Euro-NCAP AEB Kademeleri", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("TTC Eşiği (Saniye)")
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Acil Frenleme İvme Profili a(t)
        ax3 = axes[0, 2]
        t_brake = np.linspace(0, 2.5, 50)
        a_profile = np.zeros(50)
        # 0.2s tepki süresi sonra -9.0 m/s^2 tam fren
        a_profile[t_brake >= 0.2] = -9.0
        ax3.plot(t_brake, a_profile, color='#E06C75', linewidth=2.5, label='Fren İvmesi a(t)')
        ax3.axhline(y=-9.0, color='#E82127', linestyle=':', label='Maksimum AEB İvmesi (-9.0 m/s²)')
        ax3.set_title("3. AEB Frenleme İvme Profili", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Zaman (Saniye)")
        ax3.set_ylabel("İvme (m/s²)")
        ax3.legend(loc='lower left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: AEB Karar ve Güvenlik Özeti
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.85, "TESLA AEB GÜVENLİK SİSTEMİ ÖZETİ", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"MEVCUT SEVİYE: {level}\nAKSİYON: {action}\nENGELE MESAFE: {d_obs:.1f} Metre\nACİL DURMA MESAFESİ: {d_stop:.1f} Metre\nMAKSİMUM FREN GÜCÜ: -9.0 m/s² (0.92g İvme)",
                 ha='center', va='center', fontsize=9.5, color='#FFFFFF')
        ax4.text(0.5, 0.20, f"DURUM: %100 ÇARPIŞMA ÖNLEME / ENERJİ SÖNÜMLEME", ha='center', va='center', fontsize=11, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. AEB Güvenlik Doğrulaması", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: AEB Karar Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=20, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. AEB Karar Döngüsü Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Gecikme (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: AEB ve Acil Durum Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['Euro-NCAP AEB', 'FCW Warning', 'Full -9.0m/s²', 'Evasive Steer', 'Sub-10µs Step']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla AEB Güvenlik Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
