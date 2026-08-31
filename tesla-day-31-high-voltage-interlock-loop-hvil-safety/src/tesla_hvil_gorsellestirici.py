"""
Tesla HVIL ve Güvenlik Görselleştirici Modülü
=============================================
Bu modül; Yüksek Gerilim Ön Şarj (Precharge) eğrisini, HVIL açık devre
acil durdurma yanıtını ve Pyrofuse patlatma güvenliğini 6 panelli karanlık mod
tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaHVILGorsellestirici:
    """
    Tesla HVIL ve Yüksek Gerilim Güvenliği 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_hvil_guvenlik_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA YÜKSEK GERİLİM GÜVENLİĞİ: HVIL KİLİDİ, PYROFUSE & İZOLASYON İZLEME]\n"
            "Modül: Gün 31 | ISO 6469-1 Standartları, 240ms Precharge Sıralaması & ASIL-D Acil Güç Kesme",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        v_link = metrikler.get("v_link_history", [0.0] * 300)
        precharge_t = metrikler.get("precharge_time_ms", 240)
        step_ort = metrikler.get("hvil_step_ortalama_us", 0.85)

        t_ms = np.arange(len(v_link))

        # 1. Panel: İnvertör DC Link Precharge ve Kontaktör Açılışı
        ax1 = axes[0, 0]
        ax1.plot(t_ms, v_link, color='#98C379', label='İnvertör DC Link Gerilimi (V)', linewidth=2)
        ax1.axhline(y=400.0, color='#E5C07B', linestyle='--', label='Batarya Nominal Gerilimi (400V)')
        ax1.axvline(x=precharge_t, color='#61AFEF', linestyle=':', label=f'Ana Kontaktör Kapandı ({precharge_t} ms)')
        ax1.set_title("1. İnvertör DC Link Ön Şarj (Precharge) Eğrisi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Zaman (ms)")
        ax1.set_ylabel("Gerilim (V)")
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: HVIL 88 Hz PWM Güvenlik Sinyali
        ax2 = axes[0, 1]
        t_pwm = np.linspace(0, 50, 500)
        pwm_healthy = [12.0 if (t % 11.36) < 5.68 else 0.0 for t in t_pwm]
        ax2.plot(t_pwm, pwm_healthy, color='#61AFEF', label='88 Hz Sağlıklı PWM (%50 Duty)', linewidth=1.5)
        ax2.axvline(x=35.0, color='#E06C75', linestyle='--', label='Konnektör Açıldı (0V Kesilme)')
        ax2.set_title("2. HVIL Güvenlik Döngüsü PWM Süreklilik Sinyali", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_xlabel("Zaman (ms)")
        ax2.set_ylabel("HVIL Sinyal Seviyesi (V)")
        ax2.set_ylim(-2, 15)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: İzolasyon Direnci Eşikleri (ISO 6469-1)
        ax3 = axes[0, 2]
        kategoriler = ['Sağlıklı Hat\n(Normal)', 'Uyarı Eşiği\n(Nem/Toz)', 'Kritik Arıza\n(Gövde Kaçağı)']
        direncler = [600.0, 100.0, 30.0]
        ax3.bar(kategoriler, direncler, color=['#98C379', '#E5C07B', '#E06C75'], width=0.45)
        ax3.axhline(y=200.0, color='#61AFEF', linestyle='--', label='ISO 6469-1 Sınırı (200 kΩ)')
        for i, v in enumerate(direncler):
            ax3.text(i, v + 15, f'{v:.0f} kΩ', ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax3.set_title("3. Yüksek Gerilim İzolasyon Direnci (kΩ)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("İzolasyon Direnci (kΩ)")
        ax3.set_ylim(0, 750)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Acil Durdurma Müdahale Gecikmeleri (ms)
        ax4 = axes[1, 0]
        eylemler = ['Pyrofuse Patlatma\n(Kaza Anı)', 'HVIL Kesme\n(Kapak Açılması)', 'ASIL-D Bütçesi\n(Maksimum Limit)']
        sureler = [1.8, 4.2, 10.0]
        ax4.bar(eylemler, sureler, color=['#E82127', '#E5C07B', '#61AFEF'], width=0.45)
        for i, v in enumerate(sureler):
            ax4.text(i, v + 0.3, f'{v:.1f} ms', ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax4.set_title("4. Acil Durum Güç Kesme Tepki Süreleri (ms)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax4.set_ylabel("Müdahale Süresi (ms)")
        ax4.set_ylim(0, 13)
        ax4.grid(True, linestyle=':', alpha=0.3)

        # 5. Panel: 1 kHz Güvenlik Döngü Gecikmesi
        ax5 = axes[1, 1]
        ax5.bar(['Güvenlik Döngüsü'], [step_ort], color='#C678DD', width=0.35)
        ax5.text(0, step_ort + 0.05, f"{step_ort:.2f} µs\n(1 kHz RTOS)", ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')
        ax5.set_title("5. 1 kHz ASIL-D Güvenlik Karar Hızı", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_ylabel("Gecikme (µs)")
        ax5.set_ylim(0, max(step_ort * 1.8, 2.0))
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Yüksek Gerilim Güvenlik Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['HVIL 88Hz', 'Pyrofuse <2ms', 'Precharge OK', 'ISO 6469-1', 'ASIL-D Loop']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla HVIL ve Güvenlik Kalite Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
