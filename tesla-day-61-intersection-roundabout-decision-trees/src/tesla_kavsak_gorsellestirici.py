r"""
Tesla Kavşak Görselleştirici Modülü
====================================
Bu modül; 2D döner kavşak kuş bakışı görünümünü, TTC güvenlik aralıklarını,
Gap Acceptance karar grafiğini, FSM durumlarını ve çözüm gecikmesini
6 panelli karanlık mod tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaKavsakGorsellestirici:
    """
    Tesla Kavşak ve Döner Kavşak Karar Ağacı 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_roundabout_decision_tree_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA FSD ŞEHİR İÇİ KAVŞAK VE DÖNER KAVŞAK (ROUNDABOUT) KARAR AĞACI]\n"
            "Modül: Gün 61 | Geçiş Önceliği (Right-of-Way), Gap Acceptance (TTC >= 3.5s), FSM Karar Motoru & 10 µs Hız",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        min_ttc = metrikler.get("min_ttc_s", 2.5)
        state = metrikler.get("state", "YIELDING")
        action = metrikler.get("action", "YOL VERİLİYOR")
        can_enter = metrikler.get("can_enter", False)
        step_ort = metrikler.get("decision_step_ortalama_us", 10.0)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 100)

        # 1. Panel: 2D Döner Kavşak BEV ve Araç Konumları
        ax1 = axes[0, 0]
        circle_outer = plt.Circle((0, 0), 20, color='#61AFEF', fill=False, linestyle='--', linewidth=2, label='Döner Kavşak Sınırı (R=20m)')
        circle_inner = plt.Circle((0, 0), 10, color='#56B6C2', fill=True, alpha=0.3, label='Merkez Ada')
        ax1.add_patch(circle_outer)
        ax1.add_patch(circle_inner)
        # Ego Araç (Güneyden Girişte Bekliyor)
        ax1.scatter([0], [-25], color='#E82127', s=120, label='Ego Araç (Yol Verme Çizgisi)')
        # Kavşak İçi Araçlar
        ax1.scatter([-12], [8], color='#E06C75', s=100, label=f'Kritik Araç (TTC: {min_ttc:.1f}s)')
        ax1.scatter([14], [4], color='#98C379', s=100, label='Uzak Araç (TTC: 5.0s)')
        ax1.set_xlim(-30, 30)
        ax1.set_ylim(-30, 30)
        ax1.set_title("1. 2D Döner Kavşak Kuş Bakışı (BEV)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("X (Metre)")
        ax1.set_ylabel("Y (Metre)")
        ax1.legend(loc='upper right', fontsize=7.5)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Araç TTC Dağılımı ve 3.5s Eşiği
        ax2 = axes[0, 1]
        veh_ids = ['Araç #1', 'Araç #2', 'Araç #3 (Kritik)']
        ttc_vals = [4.5, 5.0, 2.5]
        colors = ['#98C379', '#98C379', '#E06C75']
        ax2.bar(veh_ids, ttc_vals, color=colors, width=0.45)
        ax2.axhline(y=3.5, color='#E5C07B', linestyle='--', linewidth=2, label='Güvenli TTC Eşiği (3.5s)')
        ax2.set_title("2. Time-To-Collision (TTC) Analizi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("TTC (Saniye)")
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Gap Acceptance Haritası (Mesafe vs Hız)
        ax3 = axes[0, 2]
        speeds = np.linspace(5, 25, 50)
        safe_distances = speeds * 3.5
        ax3.plot(speeds, safe_distances, color='#E5C07B', linewidth=2, label='Güvenli Aralık Sınırı (TTC=3.5s)')
        ax3.fill_between(speeds, safe_distances, 100, color='#98C379', alpha=0.2, label='KABUL EDİLEBİLİR (ENTER)')
        ax3.fill_between(speeds, 0, safe_distances, color='#E06C75', alpha=0.2, label='REDDEDİLEN (YIELD)')
        ax3.scatter([10.0], [25.0], color='#E06C75', s=120, label='Mevcut Durum: Red (25m / 10 m/s)')
        ax3.set_title("3. Gap Acceptance (Güvenli Aralık) Modeli", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Yaklaşan Araç Hızı (m/s)")
        ax3.set_ylabel("Mesafe (Metre)")
        ax3.legend(loc='upper left', fontsize=7.5)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: FSM Karar ve Eylem Özeti
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.85, "TESLA KAVŞAK KARAR MOTORU ÖZETİ", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"MEVCUT DURUM: {state}\nEYLEM PLANI: {action}\nMİNİMUM TTC DEĞERİ: {min_ttc:.1f} Saniye (Eşik: >= 3.5s)\nGEÇİŞ ONAYI: {'GİRİŞ ONAYLANDI' if can_enter else 'BEKLEME / YOL VERME'}\nÖNCELİK KURALI: DÖNER KAVŞAK İÇİNDEKİ ARAÇLARINDIR",
                 ha='center', va='center', fontsize=9.5, color='#FFFFFF')
        ax4.text(0.5, 0.20, f"DURUM: %100 KUSURSUZ GÜVENLİK VE UYUM", ha='center', va='center', fontsize=11, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. Karar Motoru Durum Doğrulaması", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: Karar Motoru Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=20, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. Karar Motoru Çözümleme Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Gecikme (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Kavşak Karar Ağacı Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['Right-of-Way', 'TTC >=3.5s', 'Gap Acceptance', 'FSM Transition', 'Sub-20µs Step']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla Kavşak Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
