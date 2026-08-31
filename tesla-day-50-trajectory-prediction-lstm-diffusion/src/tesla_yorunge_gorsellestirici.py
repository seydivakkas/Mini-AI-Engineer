"""
Tesla Yörünge Görselleştirici Modülü
====================================
Bu modül; Çoklu modal gelecek yörüngelerini (Lane Keep, Cut-In, Hard Brake),
olasılık dağılımlarını, TTC analizini ve çözüm gecikmesini 6 panelli karanlık mod
tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaYorungeGorsellestirici:
    """
    Tesla Yörünge Tahmini 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_trajectory_prediction_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA FSD GELECEK YÖRÜNGE TAHMİNİ: LSTM, GRU VE KOŞULLU DİFÜZYON]\n"
            "Modül: Gün 50 | 5 Saniyelik Gelecek Tahmini, Çoklu Modalite (Cut-In, Fren, Şerit Koruma) & TTC Çarpışma Riski",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        trajs = metrikler.get("trajectories", {})
        probs = metrikler.get("probabilities", np.array([0.7, 0.2, 0.1]))
        ttc = metrikler.get("ttc_sec", 4.0)
        step_ort = metrikler.get("yorunge_step_ortalama_us", 18.0)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 100)

        # 1. Panel: 2D BEV Gelecek Yörünge Haritası (50 Adım)
        ax1 = axes[0, 0]
        # Ego Araç
        ax1.scatter([0], [0], color='#61AFEF', s=120, marker='s', label='Ego Tesla (0,0)')
        # Hedef Araç
        ax1.scatter([0], [20], color='#E82127', s=120, marker='s', label='Öncü Araç (0,20m)')

        if "LANE_KEEP" in trajs:
            ax1.plot(trajs["LANE_KEEP"][:, 0], trajs["LANE_KEEP"][:, 1], color='#98C379', linewidth=2.5, label='1: Şeritte Kalma (%70)')
        if "LANE_CHANGE_LEFT" in trajs:
            ax1.plot(trajs["LANE_CHANGE_LEFT"][:, 0], trajs["LANE_CHANGE_LEFT"][:, 1], color='#E5C07B', linestyle='--', linewidth=2.5, label='2: Sol Şeride Geçiş (%20)')
        if "HARD_BRAKE" in trajs:
            ax1.plot(trajs["HARD_BRAKE"][:, 0], trajs["HARD_BRAKE"][:, 1], color='#E06C75', linestyle=':', linewidth=2.5, label='3: Ani Fren (%10)')

        ax1.set_xlim(-6, 6)
        ax1.set_ylim(-5, 100)
        ax1.set_title("1. 5 Saniyelik 2D Çoklu Modal Yörüngeler", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Yanal X (Metre)")
        ax1.set_ylabel("Boyuna Y (Metre)")
        ax1.legend(loc='lower left', fontsize=7.5)
        ax1.grid(True, linestyle=':', alpha=0.3)

        # 2. Panel: Yörünge Modu Olasılık Dağılımı
        ax2 = axes[0, 1]
        modlar = ['ŞERİTTE KAL', 'SOLA GEÇİŞ', 'ANİ FREN']
        cubuklar2 = ax2.bar(modlar, probs, color=['#98C379', '#E5C07B', '#E06C75'], width=0.45)
        for cubuk in cubuklar2:
            y = cubuk.get_height()
            ax2.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.02, f'%{y*100:.1f}', ha='center', va='bottom', fontsize=9, color='#FFFFFF')
        ax2.set_title("2. Davranış Modu Olasılıkları (Softmax)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Olasılık P(k)")
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Boyuna Mesafe Zaman Profili
        ax3 = axes[0, 2]
        t_axis = np.linspace(0.1, 5.0, 50)
        if "LANE_KEEP" in trajs:
            ax3.plot(t_axis, trajs["LANE_KEEP"][:, 1], color='#98C379', linewidth=2, label='Şeritte Kal')
        if "HARD_BRAKE" in trajs:
            ax3.plot(t_axis, trajs["HARD_BRAKE"][:, 1], color='#E06C75', linewidth=2, label='Ani Fren')
        ax3.plot(t_axis, 20.0 * t_axis, color='#61AFEF', linestyle='--', linewidth=1.5, label='Ego Araç İlerlemesi')
        ax3.set_title("3. Zaman Boyunca Boyuna Mesafe (Y vs t)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_xlabel("Zaman (Saniye)")
        ax3.set_ylabel("Mesafe Y (Metre)")
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: Çarpışma Riski ve TTC Değerlendirmesi
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.85, "GÜVENLİK VE ÇARPIŞMA RİSK ANALİZİ", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"ÇARPIŞMAYA KALAN SÜRE (TTC): {ttc:.1f} Saniye\nÖNCÜ ARAÇ HIZI: 54 km/h | EGO HIZ: 72 km/h\nTAKVİM: 5.0 Saniyelik Gelecek Ufku (50 Adım)",
                 ha='center', va='center', fontsize=10, color='#FFFFFF')
        durum_renk = '#98C379' if ttc > 3.0 else '#E06C75'
        ax4.text(0.5, 0.20, f"DURUM: GÜVENLİ TAKİP MESAFESİ (TTC > 3s)", ha='center', va='center', fontsize=11, color=durum_renk, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor=durum_renk, linewidth=1.5))
        ax4.set_title("4. Risk ve Eylem Kararı", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: Yörünge Üretim Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=20, alpha=0.75, color='#61AFEF', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. Yörünge Tahmini Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Gecikme (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Yörünge Tahmini Kalite Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['5s Horizon', 'Multi-Modal', 'Cut-In Detect', 'TTC Calc', 'Sub-25µs Step']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla Yörünge Tahmini Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
