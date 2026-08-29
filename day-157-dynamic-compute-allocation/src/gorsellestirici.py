"""
Dinamik Compute Tahsisi Teşhis Panosu Görselleştirici Modülü (Day 157 - Faz 8).
6 panelli Token Tüketimi, Maliyet Tasarrufu, Gecikme Kıyası, Soru Bazlı Bütçeler, Rotalama Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class DinamikComputeGorsellestirici:
    """Dinamik Compute tahsisi teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        simulasyon_sonucu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/dynamic_compute_allocation_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 157: Soru Zorluğuna Göre Dinamik Hesaplama ve Token Bütçesi Tahsisi (Dynamic Compute Routing)",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Toplam Token Tüketimi (Sabit vs Dinamik)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        modlar = ["Sabit Maks Bütçe\n(Her Soru 4096)", "Dinamik Bütçe\n(Easy/Hard Routing)"]
        tokenlar = [simulasyon_sonucu["toplam_sabit_token"], simulasyon_sonucu["toplam_dinamik_token"]]
        renkler1 = ["#e74a3b", "#1cc88a"]

        barlar1 = ax1.bar(modlar, tokenlar, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 350, f"{int(h):,} tok", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_title(f"1. Toplam Token Tüketimi (%{simulasyon_sonucu['token_tasarrufu_yuzde']} Tasarruf)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Harcanan Token Sayısı")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Toplam Çıkarım Maliyeti (TL)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        maliyetler = [simulasyon_sonucu["toplam_sabit_maliyet_tl"], simulasyon_sonucu["toplam_dinamik_maliyet_tl"]]
        barlar2 = ax2.bar(modlar, maliyetler, color=["#f6c23e", "#4e73df"], edgecolor="black", width=0.45)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f} TL", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax2.set_title(f"2. Çıkarım Maliyeti (%{simulasyon_sonucu['maliyet_tasarrufu_yuzde']} Tasarruf)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Maliyet (TL)")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Toplam Gecikme / Süre (Saniye)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        sureler = [simulasyon_sonucu["toplam_sabit_sure_ms"] / 1000.0, simulasyon_sonucu["toplam_dinamik_sure_ms"] / 1000.0]
        barlar3 = ax3.bar(modlar, sureler, color=["#e74a3b", "#36b9cc"], edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.2f} sn", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax3.set_title(f"3. Toplam Yanıt Gecikmesi ({simulasyon_sonucu['hizlanma_orani']}x Hızlanma)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Toplam Süre (Saniye)")
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Soru Bazında Dinamik Bütçe Dağılımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        soru_etiketleri = [f"Q{i+1} ({s['kategori']})" for i, s in enumerate(simulasyon_sonucu["soru_sonuclari"])]
        butceler = [s["tahsis_edilen_token_butcesi"] for s in simulasyon_sonucu["soru_sonuclari"]]
        renk_haritasi = {"Kolay": "#1cc88a", "Orta": "#f6c23e", "Zor": "#e74a3b"}
        cubuk_renkleri = [renk_haritasi[s["kategori"]] for s in simulasyon_sonucu["soru_sonuclari"]]

        barlar4 = ax4.bar(soru_etiketleri, butceler, color=cubuk_renkleri, edgecolor="black")
        for bar in barlar4:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, h + 60, f"{int(h)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax4.set_title("4. Soru Bazlı Dinamik Token Bütçeleri", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Tahsis Edilen Token")
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: Dynamic Compute Rotalama Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Dynamic Compute Routing Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         DYNAMIC COMPUTE & TOKEN ROUTING            \n"
            "====================================================\n"
            "  [Girdi Sorusu] ────────┐                          \n"
            "                         ▼                          \n"
            "            [Zorluk & Entropi Tahmincisi]           \n"
            "                         │                          \n"
            "          ┌──────────────┼──────────────┐           \n"
            "          ▼              ▼              ▼           \n"
            "     [KOLAY: %15]   [ORTA: %45]    [ZOR: %85]       \n"
            "          │              │              │           \n"
            "     System 1        Standart      Derin Arama      \n"
            "     Doğrudan         CoT Zinciri    MCTS + Ağaç     \n"
            "     32 Token        512 Token     4096 Token       \n"
            "     (40 ms)         (320 ms)      (2400 ms)        \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 157 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 157 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 157 SUMMARY: DYNAMIC COMPUTE ALLOCATION      \n"
            "====================================================\n"
            f"• Toplam Test Sorusu     : {simulasyon_sonucu['toplam_soru_sayisi']} Adet (Kolay / Orta / Zor)\n"
            f"• Token Tasarruf Oranı   : %{simulasyon_sonucu['token_tasarrufu_yuzde']}\n"
            f"• Maliyet Tasarrufu      : %{simulasyon_sonucu['maliyet_tasarrufu_yuzde']}\n"
            f"• Çıkarım Hızlanması     : {simulasyon_sonucu['hizlanma_orani']}x Kat Hızlı\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Triviyal sorularda binlerce gereksiz token yakılmaması\n"
            "  2. Zorlu matematik/kod problemlerine maksimum bütçe ayrılması\n"
            "  3. Akıllı çıkarım rotalama ile Pareto optimal maliyet/gecikme\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 158 (Reasoning Trace Distillation - R1)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.8,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Dinamik Compute Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
