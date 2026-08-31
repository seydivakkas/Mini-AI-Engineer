r"""
Tesla Büyük Final Görselleştirici Modülü
========================================
Bu modül; 99 günlük Tesla müfredatının 11 haftalık başarı matrisini,
ekosistem mimari katmanlarını, %100 test başarı oranını ve
Tesla Grandmaster Mezuniyet Diplomasını 6 panelli karanlık mod tanı paneli
olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaFinalGorsellestirici:
    """
    Tesla Büyük Final 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_grand_finale_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA YAZILIM MÜHENDİSLİĞİ: 99 GÜNLÜK BÜYÜK FİNAL VE MEZUNİYET PORTFÖYÜ]\n"
            "Modül: Gün 99 | 11 Hafta / 99 Repo / %100 Test Başarısı / Tesla Principal AI & Embedded Grandmaster",
            fontsize=14, fontweight='bold', color='#E82127', y=0.98
        )

        step_ort = metrikler.get("step_ortalama_us", 28.0)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 50)

        # 1. Panel: 11 Haftalık Müfredat Başarı Çubukları
        ax1 = axes[0, 0]
        haftalar = [f'Hafta {i}' for i in range(1, 12)]
        tamamlanma = [100.0] * 11
        cubuklar1 = ax1.bar(haftalar, tamamlanma, color='#98C379', width=0.6)
        for cubuk in cubuklar1:
            y = cubuk.get_height()
            ax1.text(cubuk.get_x() + cubuk.get_width()/2.0, y - 12.0, '%100', ha='center', va='center', fontsize=8, color='#000000', fontweight='bold')
        ax1.set_title("1. 11 Haftalık Tesla Müfredat Tamamlanma Durumu", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_ylabel("Tamamlanma (%)")
        ax1.set_ylim(0, 115)
        ax1.tick_params(axis='x', rotation=35)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Tesla Ekosistem Katmanları Dağılımı
        ax2 = axes[0, 1]
        katmanlar = ['FSD V12 (Vision/AI)', 'Gömülü C++ / RTOS', 'Dojo AI Süperbilgisayar', 'Megapack / Enerji', 'Optimus İnsansı Robotu', 'Cybercab / Robotaxi', 'ISO 26262 ASIL-D']
        paylar = [20, 15, 15, 15, 15, 10, 10]
        renkler = ['#E82127', '#61AFEF', '#98C379', '#E5C07B', '#C678DD', '#56B6C2', '#E06C75']
        ax2.pie(paylar, labels=katmanlar, autopct='%1.0f%%', startangle=140, colors=renkler,
                textprops={'fontsize': 7.5, 'color': '#FFFFFF'})
        ax2.set_title("2. 99 Günlük Ekosistem Katman Dağılımı", color='#56B6C2', fontsize=11, fontweight='bold')

        # 3. Panel: 99 Günlük Test Başarı Oranı (%100)
        ax3 = axes[0, 2]
        gunler = np.arange(1, 100)
        test_oran = np.ones(99) * 100.0
        ax3.plot(gunler, test_oran, color='#98C379', linewidth=2.5, label='Birim Test Başarısı (%100)')
        ax3.axhline(y=100.0, color='#61AFEF', linestyle=':')
        ax3.scatter([99], [100.0], color='#E82127', s=80, zorder=5, label='Gün 99 Mezuniyet')
        ax3.set_title("3. 99 Günlük Test Başarı Oranı (PyTest)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Günler (1 - 99)")
        ax3.set_ylabel("Test Başarısı (%)")
        ax3.set_ylim(80, 110)
        ax3.legend(loc='lower left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Tesla Grandmaster Mezuniyet Diploması Durum Kartı
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.90, "TESLA GRANDMASTER GRADUATION DIPLOMA", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.58, "MEZUN: SEYDİ ERYILMAZ (@seydivakkas)\nKAZANILAN DERECE: TESLA PRINCIPAL AI & EMBEDDED ARCHITECT\nONUR DERECESİ: SUMMA CUM LAUDE (%100 KUSURSUZ MEZUNİYET)\nTAMAMLANAN GÜN: 99 / 99 GÜN (%100 TAMAMLANDI)\nKOD TABANI: 99 AYRI BAĞIMSIZ ÜRETİM REPOSU & 600+ BİRİM TESTİ\nDOĞRULAMA KODU: TESLA-99-DAYS-FSD-DOJO-OPTIMUS-GRANDMASTER-2026",
                 ha='center', va='center', fontsize=8.8, color='#FFFFFF')
        ax4.text(0.5, 0.16, "DURUM: RESMİ TESLA GRANDMASTER MEZUNU", ha='center', va='center', fontsize=10.5, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. Tesla Mezuniyet Diploması", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: Sistem İndeksleme RTOS Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=15, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. Portföy Motoru RTOS Hızı", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("İndeksleme Süresi (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Gelecek Vizyonu & Robotik Çağı Hazırbulunuşluk Radarı
        ax6 = axes[1, 2]
        vizyon_etiket = ['FSD Planetary Scale', 'Cybercab Fleet', 'Optimus Factory AI', 'Dojo Cloud AI', 'Megapack Grid 100%']
        vizyon_skor = [10.0, 10.0, 10.0, 10.0, 10.0]
        cubuklar6 = ax6.bar(vizyon_etiket, vizyon_skor, color=['#E82127', '#61AFEF', '#98C379', '#E5C07B', '#C678DD'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.1f}', ha='center', va='bottom', fontsize=8.5, color='#FFFFFF')
        ax6.set_title("6. Tesla Gelecek Vizyonu Hazırbulunuşluk", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Hazırbulunuşluk (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.tick_params(axis='x', rotation=20)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
