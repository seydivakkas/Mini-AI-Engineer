"""
Nedensellik ve Do-Calculus Teşhis Panosu Görselleştirici Modülü (Day 159 - Faz 8).
6 panelli Pearl Merdiveni, Korelasyon vs ATE, Karşıgelişçi Analiz, Causal DAG, Do-Calculus Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class NedensellikGorsellestirici:
    """Nedensellik ve Causal Inference teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        gozlem: Dict[str, float],
        mudahale: Dict[str, float],
        karsigelisci: Dict[str, Any],
        kayit_yolu: str = "ciktilar/causal_reasoning_dag_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 159: Nedensellik Analizi (Causal Inference & Reasoning): Causal DAG, Do-Calculus & Karşıgelişçi Akıl Yürütme",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Judea Pearl Nedensellik Merdiveni (3 Seviye)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        basamaklar = ["L3: Karşıgelişçi\n(Counterfactuals)\n'Ya almasaydı?'", "L2: Müdahale\n(Intervention)\n'do(X=1)'", "L1: Gözlem / Korelasyon\n(Association)\n'P(Y|X)'"]
        degerler1 = [100.0, 70.0, 40.0]
        renkler1 = ["#e74a3b", "#4e73df", "#1cc88a"]

        barlar1 = ax1.barh(basamaklar, degerler1, color=renkler1, edgecolor="black", height=0.5)
        for bar in barlar1:
            w = bar.get_width()
            ax1.text(w + 2.0, bar.get_y() + bar.get_height() / 2, f"Seviye {int(w/30)+1}", ha="left", va="center", fontsize=10, fontweight="bold")

        ax1.set_title("1. Judea Pearl Nedensellik Merdiveni", fontsize=12, fontweight="bold")
        ax1.set_xlim(0, 125)
        ax1.grid(axis="x", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Gözlemsel Korelasyon vs Gerçek Nedensel Etki (ATE)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        metrikler = ["Gözlemsel Fark\n(P(Y|X=1) - P(Y|X=0))\n(Şişirilmiş)", "Do-Calculus ATE\n(P(Y|do(X=1)) - P(Y|do(X=0)))\n(Gerçek Etki)"]
        etkiler = [gozlem["gozlemsel_fark"] * 100.0, mudahale["ortalama_nedensel_etki_ate"] * 100.0]

        barlar2 = ax2.bar(metrikler, etkiler, color=["#f6c23e", "#1cc88a"], edgecolor="black", width=0.45)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"+%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax2.set_title("2. Sahte Korelasyon vs Do-Calculus Gerçek ATE", fontsize=12, fontweight="bold")
        ax2.set_ylabel("İyileşme Artışı (%)")
        ax2.set_ylim(0, 45)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Bireysel Karşıgelişçi Analiz
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        durumlar = ["Gerçekleşen\n(İlaç Aldı & İyileşti)", "Karşıgelişçi\n(İlaç Almasaydı Ne Olurdu?)"]
        oranlar = [90.0, karsigelisci["karsigelisci_iyilesme_olasiligi"] * 100.0]

        barlar3 = ax3.bar(durumlar, oranlar, color=["#4e73df", "#36b9cc"], edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax3.set_title(f"3. Karşıgelişçi İyileşme (PN: %{karsigelisci['zorunluluk_olasiligi_pn']*100:.1f})", fontsize=12, fontweight="bold")
        ax3.set_ylabel("İyileşme İhtimali (%)")
        ax3.set_ylim(0, 110)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Causal DAG Görseli ve Yapısı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Neden-Sonuç Grafı (Causal DAG)", fontsize=12, fontweight="bold", pad=10)

        dag_metni = (
            "====================================================\n"
            "        CAUSAL DIRECTED ACYCLIC GRAPH (DAG)        \n"
            "====================================================\n"
            "                  [Z: Yaş Grubu]                    \n"
            "                  (Konfondör)                       \n"
            "                    /       \\                       \n"
            "         (Backdoor)/         \\                      \n"
            "                  v           v                     \n"
            "            [X: İlaç] ──────> [Y: İyileşme]         \n"
            "           (Müdahale) (Saf    (Sonuç)               \n"
            "                       Etki)                        \n"
            "----------------------------------------------------\n"
            "  • Düğümler (Nodes): {Z: Yaş, X: İlaç, Y: İyileşme}\n"
            "  • Backdoor Yolu   : X <- Z -> Y (Sahte İlişki)    \n"
            "  • Do(X) Müdahalesi: Z -> X bağı kesilir!          \n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, dag_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Do-Calculus ve Backdoor Formülü
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Do-Calculus & Backdoor Adjustment Formülü", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "====================================================\n"
            "            DO-CALCULUS MATHEMATICAL ENGINE         \n"
            "====================================================\n"
            "  Backdoor Ayarlama Formülü:                        \n"
            "    P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) · P(Z=z)  \n"
            "----------------------------------------------------\n"
            "  Hesaplama Adımları:                               \n"
            "  • P(Y=1 | do(X=1)) = 0.90*(0.5) + 0.50*(0.5) = 0.70\n"
            "  • P(Y=1 | do(X=0)) = 0.80*(0.5) + 0.40*(0.5) = 0.60\n"
            "  • Ortalama Nedensel Etki (ATE):                   \n"
            "      ATE = 0.70 - 0.60 = +0.10 (+%10.0)            \n"
            "----------------------------------------------------\n"
            "  SONUÇ: Yaş konfondörü nötrlenerek ilacın net      \n"
            "         faydasının +%10 olduğu kanıtlandı!         \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, formuller,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 159 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 159 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 159 SUMMARY: CAUSAL REASONING & DO-CALCULUS  \n"
            "====================================================\n"
            "• Gözlemsel Korelasyon : +%34.0 (Yanıltıcı Şişirilmiş)\n"
            "• Do-Calculus ATE      : +%10.0 (Gerçek Nedensel Fayda)\n"
            "• Karşıgelişçi Durum   : İlaç almasaydı %80 iyileşirdi\n"
            "• Zorunluluk Olasılığı : %11.1 (İyileşme ilaca bağlıydı)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Korelasyon ile Nedensellik arasındaki ayrım\n"
            "  2. Causal DAG üzerinde Backdoor yolunu bloke etme\n"
            "  3. P(Y|do(X)) ile aktif müdahale hesaplama\n"
            "  4. 3. Seviye Karşıgelişçi 'Ya öyle olmasaydı?' analizi\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 160 (FAZ 8 BÜYÜK FİNALİ - Benchmark Suite)\n"
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
        print(f"  ✓ Nedensellik Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
