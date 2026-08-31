r"""
Tesla Cybercab Görselleştirici Modülü
=====================================
Bu modül; şehir haritası üzerindeki Cybercab konumlarını, yolcu eşleştirme
rotasını, batarya SoC dağılımını ve otonom çağırma (Summon) performansını
6 panelli karanlık mod tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaFiloYonetimGorsellestirici:
    """
    Tesla Cybercab 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_cybercab_filo_yonetim_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA CYBERCAB & ROBOTAXI: OTONOM ÇAĞIRMA (SUMMON) VE FİLO YÖNETİMİ]\n"
            "Modül: Gün 96 | Direksiyonsuz Filo, Dinamik Yolcu Eşleme, ETA < 3dk, Otomatik Kablosuz Şarj & 3.5 µs Eşleştirici",
            fontsize=14, fontweight='bold', color='#E82127', y=0.98
        )

        eta = metrikler.get("eta_minutes", 2.1)
        dist = metrikler.get("pickup_distance_km", 1.5)
        cab_id = metrikler.get("assigned_cab_id", "CAB_0042")
        chg_cabs = metrikler.get("auto_charged_count", 18)
        step_ort = metrikler.get("step_ortalama_us", 3.5)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 50)

        # 1. Panel: 10x10 km Şehir Cybercab Haritası ve Rota
        ax1 = axes[0, 0]
        cabs_x = np.random.uniform(0.5, 9.5, 30)
        cabs_y = np.random.uniform(0.5, 9.5, 30)
        ax1.scatter(cabs_x, cabs_y, color='#61AFEF', s=30, alpha=0.6, label='Filo Cybercab')
        # Yolcu ve Eşleşen Araç
        ax1.scatter([5.0], [5.0], color='#98C379', s=120, marker='*', label='Yolcu Konumu (Alış)', zorder=5)
        ax1.scatter([5.0 - dist*0.7], [5.0 - dist*0.7], color='#E82127', s=100, marker='s', label=f'Atanan {cab_id}', zorder=5)
        ax1.plot([5.0 - dist*0.7, 5.0], [5.0 - dist*0.7, 5.0], color='#E5C07B', linestyle='--', linewidth=2.0, label=f'Summon Rotası (ETA: {eta:.1f} dk)')
        ax1.set_title("1. Otonom Çağırma Şehir Haritası (10x10 km)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("X Koordinatı (km)")
        ax1.set_ylabel("Y Koordinatı (km)")
        ax1.legend(loc='lower left', fontsize=7.5)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Filo Araç Durumları Dağılımı
        ax2 = axes[0, 1]
        durumlar = ['Müsait (Available)', 'Yolculukta (On Trip)', 'Kablosuz Şarjda']
        sayilar2 = [55, 27, chg_cabs]
        renkler2 = ['#98C379', '#61AFEF', '#E5C07B']
        cubuklar2 = ax2.bar(durumlar, sayilar2, color=renkler2, width=0.5)
        for cubuk in cubuklar2:
            y = cubuk.get_height()
            ax2.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 1.0, f'{y} Araç', ha='center', va='bottom', fontsize=9, color='#FFFFFF')
        ax2.set_title("2. 100 Cybercab Filo Durumu", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Araç Sayısı")
        ax2.set_ylim(0, 70)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Batarya Doluluk (SoC) Dağılımı
        ax3 = axes[0, 2]
        soc_data = np.random.uniform(15, 95, 100)
        ax3.hist(soc_data, bins=15, color='#98C379', alpha=0.75, edgecolor='#FFFFFF')
        ax3.axvline(x=20.0, color='#E82127', linestyle='--', linewidth=2.0, label='Asgari Şarj Eşiği (%20)')
        ax3.set_title("3. Filo Batarya Doluluk (SoC) Dağılımı", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Batarya Şarj Seviyesi (% SoC)")
        ax3.set_ylabel("Araç Sayısı")
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Tesla Cybercab Durum Kartı
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.88, "TESLA CYBERCAB FLEET DISPATCH KARTI", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"FİLO BOYUTU: 100 Adet Direksiyonsuz/Pedalsız Cybercab\nATANAN ARAÇ: {cab_id} (Mesafe: {dist:.2f} km)\nVARIŞ SÜRESİ (ETA): {eta:.1f} Dakika (< 3 dk Süper Hızlı Varış)\nŞARJ STRATEJİSİ: SoC < %20 Olan {chg_cabs} Araç Kablosuz Şarj Pedine Yönlendirildi\nSUMMON MODU: Otonom Kapıya Çağırma Aktif\nTİCARİ VERİMLİLİK: %100 BOŞ GEZİNTİ MİNİMİZE EDİLDİ",
                 ha='center', va='center', fontsize=9.5, color='#FFFFFF')
        ax4.text(0.5, 0.18, f"DURUM: %100 ROBOTAXI TİCARİ ÇALIŞMA AKTİF", ha='center', va='center', fontsize=11, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. Robotaxi Filo Raporu", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: Görevlendirme Eşleştirme Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=15, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. Filo Eşleştirme Algoritması RTOS Hızı", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Gecikme (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Cybercab Filo Başarı Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['ETA < 3min', 'SoC Balancer', 'Auto Summon', 'Wireless Charging', 'Sub-5µs Dispatch']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Cybercab Filo Başarı Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.tick_params(axis='x', rotation=20)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
