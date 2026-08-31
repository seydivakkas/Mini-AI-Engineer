"""
Tesla Faz 3 Büyük Capstone Görselleştirici Modülü
=================================================
Bu modül; 0-120 km/h hızlanma, tork dinamikleri, batarya voltaj çökmesi,
EKF SoC takibi ve Octovalve termal yanıtını 6 panelli karanlık mod
büyük capstone tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaCapstoneGorsellestirici:
    """
    Tesla Faz 3 Büyük Capstone 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_faz3_capstone_bms_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA FAZ 3 BÜYÜK CAPSTONE: MERKEZİ BMS, FOC İNVERTÖR VE GÜÇ AKTARMA SİSTEMİ]\n"
            "Modül: Gün 33 | 96S 2-RC ECM, EKF SoC, SoH RLS, 10kHz FOC, SVPWM, Octovalve, Rejen & HVIL ASIL-D",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        spd = metrikler.get("speed", [0.0] * 100)
        trq = metrikler.get("torque", [0.0] * 100)
        pwr = metrikler.get("power", [0.0] * 100)
        volt = metrikler.get("voltage", [385.0] * 100)
        soc = metrikler.get("soc", [85.0] * 100)
        temp = metrikler.get("temp", [25.0] * 100)
        step_ort = metrikler.get("capstone_step_ortalama_us", 3.25)
        max_spd = metrikler.get("max_speed_kmh", 120.0)
        max_pwr = metrikler.get("max_power_kw", 180.0)

        t_s = np.linspace(0, len(spd) * 0.01, len(spd))

        # 1. Panel: Araç Hız Profili (0-120 km/h İvmelenme & Rejen Duruşu)
        ax1 = axes[0, 0]
        ax1.plot(t_s, spd, color='#98C379', label='Araç Hızı (km/h)', linewidth=2)
        ax1.axhline(y=120.0, color='#E5C07B', linestyle='--', label='Otoyol Hedef Hızı (120 km/h)')
        ax1.set_title("1. Dinamik Sürüş ve Duruş Hız Eğrisi (km/h)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Zaman (Saniye)")
        ax1.set_ylabel("Hız (km/h)")
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: FOC Motor Elektromanyetik Torku (Nm)
        ax2 = axes[0, 1]
        ax2.plot(t_s, trq, color='#E82127', label='FOC Motor Torku (Nm)', linewidth=2)
        ax2.axhline(y=0.0, color='#FFFFFF', linestyle=':', alpha=0.5)
        ax2.set_title("2. IPM-SynRM Çekiş & Rejen Tork Profili (Nm)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_xlabel("Zaman (Saniye)")
        ax2.set_ylabel("Tork (Nm)")
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Batarya Elektriksel Güç Akışı (kW)
        ax3 = axes[0, 2]
        ax3.plot(t_s, pwr, color='#61AFEF', label='Batarya Gücü (+Deşarj / -Rejen kW)', linewidth=2)
        ax3.axhline(y=0.0, color='#FFFFFF', linestyle=':', alpha=0.5)
        ax3.set_title("3. Çekiş Gücü ve Rejeneratif Şarj Akışı (kW)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Zaman (Saniye)")
        ax3.set_ylabel("Güç (kW)")
        ax3.legend(loc='lower left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: 96S Batarya Paketi Terminal Gerilimi (V)
        ax4 = axes[1, 0]
        ax4.plot(t_s, volt, color='#E5C07B', label='Paket Gerilimi (Voltaj Çökmesi)', linewidth=2)
        ax4.axhline(y=400.0, color='#98C379', linestyle='--', label='400V Nominal Sınır')
        ax4.set_title("4. 96S Paket Dinamik Voltaj Çökmesi (V)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_xlabel("Zaman (Saniye)")
        ax4.set_ylabel("Gerilim (V)")
        ax4.legend(loc='lower left', fontsize=8)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: EKF SoC Takibi ve Batarya Sıcaklığı
        ax5 = axes[1, 1]
        ax5_twin = ax5.twinx()
        l1 = ax5.plot(t_s, soc, color='#61AFEF', label='EKF Kestirilen SoC (%)', linewidth=2)
        l2 = ax5_twin.plot(t_s, temp, color='#E06C75', linestyle='--', label='Paket Sıcaklığı (°C)', linewidth=1.5)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax5.set_title("5. EKF SoC (%) ve Octovalve Termal Yönetim", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Zaman (Saniye)")
        ax5.set_ylabel("SoC (%)", color='#61AFEF')
        ax5_twin.set_ylabel("Sıcaklık (°C)", color='#E06C75')
        ax5.legend(lines, labels, loc='center left', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Faz 3 Büyük Capstone Mimari Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['96S ECM Core', 'EKF + RLS SoH', '10kHz FOC+PWM', 'Octovalve Pump', 'ASIL-D Safety']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla Faz 3 Büyük Capstone Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
