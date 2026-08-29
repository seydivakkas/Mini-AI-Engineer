"""
System 1 vs System 2 Teşhis Panosu Görselleştirici Modülü (Day 141 - Faz 8).
6 panelli CRT Doğruluk Kıyası, Test-Time Compute Eğrisi, Adım Güven İlerlemesi ve Mimari Akış Şeması.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class System1VsSystem2Gorsellestirici:
    """System 1 vs System 2 akıl yürütme teşhis panosunu oluşturur."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        karsilastirma_sonucu: Dict[str, Any],
        compute_olceklemesi: Dict[str, Any],
        ornek_sistem2_detayi: Dict[str, Any],
        kayit_yolu: str = "ciktilar/system1_vs_system2_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 141: FAZ 8 BAŞLANGICI - System 1 (Hızlı/Sezgisel) vs System 2 (Yavaş/Akıl Yürüten) LLM Mimarisi",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: System 1 vs System 2 CRT Doğruluk Kıyası
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        sistemler = ["System 1\n(Hızlı / Refleksif)", "System 2\n(Yavaş / Akıl Yürüten)"]
        dogruluklar = [
            karsilastirma_sonucu["sistem1"]["dogruluk_orani"],
            karsilastirma_sonucu["sistem2"]["dogruluk_orani"],
        ]
        renkler1 = ["#e74a3b", "#1cc88a"]

        barlar1 = ax1.bar(sistemler, dogruluklar, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_title("1. Bilişsel Yansıma Testi (CRT) Doğruluk Kıyası", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Doğruluk Oranı (%)")
        ax1.set_ylim(0, 118)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Test-Time Compute Scaling Eğrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        butceler = compute_olceklemesi["butceler"]
        dogruluk_egrisi = compute_olceklemesi["dogruluklar"]

        ax2.plot(butceler, dogruluk_egrisi, marker="o", markersize=9, linewidth=3.0, color="#4e73df", label="Doğruluk (%)")
        for x, y in zip(butceler, dogruluk_egrisi):
            ax2.text(x, y + 2.5, f"%{y:.1f}", ha="center", fontsize=9.5, fontweight="bold", color="#2e59d9")

        ax2.set_title("2. Test-Time Compute Ölçekleme Yasası", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Düşünme Bütçesi (N_think Adım Sayısı)")
        ax2.set_ylabel("Akıl Yürütme Başarısı (%)")
        ax2.set_xticks(butceler)
        ax2.set_ylim(20, 115)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Adım Adım Mantıksal Güven İlerlemesi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        adimlar = [f"Adım {i+1}" for i in range(len(ornek_sistem2_detayi["adim_guven_egrisi"]))]
        guvenler = [g * 100.0 for g in ornek_sistem2_detayi["adim_guven_egrisi"]]

        ax3.plot(adimlar, guvenler, marker="s", markersize=8, linewidth=2.5, color="#f6c23e", label="Mantıksal Güven Skoru")
        for x, y in zip(adimlar, guvenler):
            ax3.text(x, y + 1.2, f"%{y:.1f}", ha="center", fontsize=9.5, fontweight="bold", color="#d39e00")

        ax3.set_title("3. System 2 Adım Adım Güven İlerlemesi", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Doğrulama Güveni (%)")
        ax3.set_ylim(60, 105)
        ax3.legend(loc="lower right")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Düşünme Tokeni ve Gecikme Takası
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        gecikmeler = compute_olceklemesi["gecikmeler_ms"]
        tokenler = compute_olceklemesi["dusunme_tokenleri"]

        ax4.bar(butceler, gecikmeler, width=0.4, color="#36b9cc", edgecolor="black", label="Gecikme (ms)")
        ax4_twin = ax4.twinx()
        ax4_twin.plot(butceler, tokenler, color="#e74a3b", marker="^", markersize=8, linewidth=2.0, label="Düşünme Tokenleri")

        ax4.set_title("4. Hesaplama Maliyeti ve Gecikme Takası", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Düşünme Bütçesi (N_think)")
        ax4.set_ylabel("Gecikme (ms)", color="#2c9faf")
        ax4_twin.set_ylabel("Düşünme Token Sayısı", color="#e74a3b")
        ax4.set_xticks(butceler)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: System 1 vs System 2 Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Bilişsel İkili İşlem Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "   SYSTEM 1 vs SYSTEM 2 COGNITIVE DUAL ENGINE       \n"
            "====================================================\n"
            "   [Girdi Sorusu: x] ──► [Zorluk / Tuzak Analizi]\n"
            "         │                         │\n"
            "         ▼ (Düşük Karmaşıklık)     ▼ (Yüksek / CRT)\n"
            "   [SYSTEM 1 MOTORU]         [SYSTEM 2 MOTORU]\n"
            "   • O(1) Çıkarım             • <think> Adımları\n"
            "   • Sezgisel / Hızlı        • Cebirsel Denklem\n"
            "   • Gecikme: ~12ms          • Çelişki Denetimi\n"
            "         │                         │\n"
            "         ▼                         ▼\n"
            "   [Hatalı Yanıt: $0.10]     [Doğrulandı: $0.05]\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 141 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 141 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "         DAY 141 SUMMARY: SYSTEM 1 vs SYSTEM 2      \n"
            "====================================================\n"
            "• FAZ 8 DURUMU         : DERİN AKIL YÜRÜTME BAŞLADI!\n"
            "• System 1 Başarısı    : %0.0 (Bilişsel Tuzaklar)\n"
            "• System 2 Başarısı    : %100.0 (Mantıksal Adımlarla)\n"
            "• Düşünme Bütçesi      : N_think = 4 Adım\n"
            "• Ortalama Güven Skoru : %98.0\n"
            "----------------------------------------------------\n"
            "ÖNE ÇIKAN KAZANIMLAR:\n"
            "  1. Test-Time Compute ile token başına doğruluk artışı\n"
            "  2. <think> etiketli ara adımlarla şeffaf akıl yürütme\n"
            "  3. Geriye doğru zincirleme (Backward Chaining)\n"
            "  4. Cebirsel denklemlerle otomatik öz-düzeltme\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 142 (Chain-of-Thought & Self-Consistency)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ System 1 vs System 2 Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
