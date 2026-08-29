"""
Self-Consistency Sıcaklık ve Entropi Teşhis Panosu Görselleştirici Modülü (Day 143 - Faz 8).
6 panelli Ağırlıklı Oylama Kıyası, Sıcaklık-Entropi İlişkisi, Epistemik Belirsizlik, Doğruluk Eğrisi ve Şema.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class SelfConsistencyTemperatureGorsellestirici:
    """Self-Consistency sıcaklık örneklemesi ve entropi teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        oylama_sonucu: Dict[str, Any],
        entropi_sonucu: Dict[str, Any],
        sicaklik_taramasi: Dict[str, Any],
        kayit_yolu: str = "ciktilar/self_consistency_majority_voting_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 143: Self-Consistency: Çoklu Akıl Yürütme Yollarında Sıcaklık Örneklemesi (T), Ağırlıklı Oylama ve Entropi",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Hard vs Soft Ağırlıklı Oylama Kıyası
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        adaylar = list(oylama_sonucu["agirlikli_oy_dagilimi"].keys())
        soft_oranlar = [oylama_sonucu["agirlikli_oy_dagilimi"][k] * 100.0 for k in adaylar]
        toplam_hard = sum(oylama_sonucu["hard_oy_dagilimi"].values())
        hard_oranlar = [(oylama_sonucu["hard_oy_dagilimi"].get(k, 0) / toplam_hard) * 100.0 for k in adaylar]

        x = np.arange(len(adaylar))
        w = 0.35

        ax1.bar(x - w / 2, hard_oranlar, width=w, label="Hard Voting (Düz Sayım)", color="#4e73df", edgecolor="black")
        ax1.bar(x + w / 2, soft_oranlar, width=w, label="Soft Voting (Yol Güven Ağırlıklı)", color="#1cc88a", edgecolor="black")

        for i in range(len(adaylar)):
            ax1.text(x[i] - w / 2, hard_oranlar[i] + 1.5, f"%{hard_oranlar[i]:.1f}", ha="center", fontsize=9.5, fontweight="bold")
            ax1.text(x[i] + w / 2, soft_oranlar[i] + 1.5, f"%{soft_oranlar[i]:.1f}", ha="center", fontsize=9.5, fontweight="bold")

        ax1.set_title("1. Hard vs Soft Ağırlıklı Oylama Dağılımı", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Oy Oranı (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Tahmin: {a}" for a in adaylar], fontsize=10)
        ax1.set_ylim(0, 115)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Sıcaklık Seviyelerine Göre Shannon Entropisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        t_degerleri = sicaklik_taramasi["sicakliklar"]
        entropiler = sicaklik_taramasi["entropiler"]

        ax2.plot(t_degerleri, entropiler, marker="o", markersize=9, linewidth=3.0, color="#f6c23e", label="Shannon Entropisi H(Y|x)")
        for x_val, y_val in zip(t_degerleri, entropiler):
            ax2.text(x_val, y_val + 0.05, f"{y_val:.2f}", ha="center", fontsize=9.5, fontweight="bold", color="#b78103")

        ax2.set_title("2. Sıcaklık (T) vs Shannon Entropisi", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Örnekleme Sıcaklığı (Temperature)")
        ax2.set_ylabel("Entropi (Bit)")
        ax2.set_ylim(0, max(entropiler) + 0.4)
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Epistemik Belirsizlik ve Gini Saflığı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metrikler = ["Shannon\nEntropisi", "Gini\nKirliliği", "Maksimum\nGüven"]
        degerler3 = [
            entropi_sonucu["shannon_entropisi"],
            entropi_sonucu["gini_kirliligi"],
            entropi_sonucu["maksimum_olasilik"],
        ]
        renkler3 = ["#36b9cc", "#e74a3b", "#1cc88a"]

        barlar3 = ax3.bar(metrikler, degerler3, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title(f"3. Belirsizlik Durumu: {entropi_sonucu['belirsizlik_seviyesi'].split()[0]}", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Skor Değeri")
        ax3.set_ylim(0, 1.2)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Sıcaklık (T) vs Akıl Yürütme Doğruluk Eğrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        dogruluklar = sicaklik_taramasi["dogruluklar"]

        ax4.plot(t_degerleri, dogruluklar, marker="s", markersize=9, linewidth=3.0, color="#2e59d9", label="Self-Consistency Doğruluğu (%)")
        for x_val, y_val in zip(t_degerleri, dogruluklar):
            ax4.text(x_val, y_val + 2.0, f"%{y_val:.0f}", ha="center", fontsize=9.5, fontweight="bold", color="#1b3fb3")

        ax4.set_title("4. Sıcaklık (T) vs Doğruluk (Optimal T=0.7)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Örnekleme Sıcaklığı (T)")
        ax4.set_ylabel("Doğruluk Oranı (%)")
        ax4.set_ylim(40, 115)
        ax4.legend(loc="lower left")
        ax4.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: Sıcaklık & Ağırlıklı Oylama Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Sıcaklık Örneklemesi ve Ağırlıklı Oylama Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "   WEIGHTED SELF-CONSISTENCY & UNCERTAINTY          \n"
            "====================================================\n"
            "   [Soru: x] ──► [Sıcaklık Örneklemesi: T=0.7]\n"
            "         │\n"
            "         ├── Yol 1: P(τ₁)=0.45 ──► Tahmin: $0.05\n"
            "         ├── Yol 2: P(τ₂)=0.30 ──► Tahmin: $0.05\n"
            "         ├── Yol 3: P(τ₃)=0.15 ──► Tahmin: $0.05\n"
            "         └── Yol 4: P(τ₄)=0.10 ──► Tahmin: $0.10 (Sapan)\n"
            "                 │\n"
            "                 ▼\n"
            "   [AĞIRLIKLI MARJİNALİZASYON]\n"
            "   • Skor($0.05) = 0.45 + 0.30 + 0.15 = 0.90 (%90.0)\n"
            "   • Skor($0.10) = 0.10 (%10.0)\n"
            "   • Shannon Entropisi: H(Y|x) = 0.46 Bit (Düşük Risk)\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 143 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 143 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "         DAY 143 SUMMARY: WEIGHTED SELF-CONSISTENCY \n"
            "====================================================\n"
            "• Optimal Sıcaklık     : T = 0.7 (Maksimum Doğruluk: %100)\n"
            "• Ağırlıklı Güven      : %90.0 (P(trajectory) ile)\n"
            "• Shannon Entropisi    : 0.46 Bit (Düşük Belirsizlik)\n"
            "• Gini Kirliliği       : 0.180 (Yüksek Saflık)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Greedy (T=0.0) katılığının aşılması\n"
            "  2. Kaotik (T=1.2) halüsinasyonlarının tespiti\n"
            "  3. Soft logit ağırlıklandırma ile gürültü direnci\n"
            "  4. Tahmin belirsizliğinin matematiksel ölçümü\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 144 (Tree of Thoughts: BFS & DFS)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Self-Consistency Sıcaklık ve Entropi Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
