"""
Structured Pruning Teşhis ve Görselleştirme Panosu
--------------------------------------------------
6 panelli yüksek çözünürlüklü yapısal filtre budama, L1/L2 norm dağılımları,
fiziksel küçültme, gecikme (Latency ms) ve doğruluk toparlanma paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


class BudamaGorsellestirici:
    """
    Structured Pruning analizlerini ve donanım hızlanma grafiklerini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_budama_paneli(
        self,
        katman_skorlari: np.ndarray,
        budama_esigi: float,
        oranlar: List[str],
        parametreler: List[int],
        gecikmeler: List[float],
        dogruluk_oncesi: List[float],
        dogruluk_sonrasi: List[float],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Structured Pruning teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 83: L1/L2 Norm Tabanlı Yapısal Filtre/Kanal Budama ve Donanım Hızlanma Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Yapısal (Structured) vs Yapısal Olmayan (Unstructured) Budama
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        pruning_metin = (
            "      STRUCTURED VS UNSTRUCTURED PRUNING MİMARİSİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. YAPISAL OLMAYAN BUDAMA (Unstructured Weight Pruning):\n"
            "     • Tek tek ağırlıklar sıfırlanır (W_ij = 0).\n"
            "     • Seyrek (Sparse) matris oluşturur.\n"
            "     • ⚠️ Standart GPU/CPU çekirdeklerinde HIZ KAZANDIRMAZ!\n\n"
            "  2. YAPISAL BUDAMA (Structured Filter Pruning - BİZİM YÖNTEM):\n"
            "     • Tüm konvolüsyonel filtre/kanal komple kesilip atılır.\n"
            "     • [C_out, C_in, K, K] ──> [C_out - k, C_in - m, K, K]\n"
            "     • Fiziksel olarak daha küçük YOĞUN (Dense) tensör oluşur.\n"
            "     • 🚀 TÜM standart donanımlarda GERÇEK HIZLANMA sağlar!"
        )
        ax1.text(
            0.5, 0.5, pruning_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Budama Tipleri ve Donanım Mekanizması", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Filtre L1 Norm Dağılımı ve Budama Eşiği
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        x_filt = np.arange(len(katman_skorlari))
        renkler = ["#e53e3e" if s < budama_esigi else "#38a169" for s in katman_skorlari]

        ax2.bar(x_filt, katman_skorlari, color=renkler, width=0.6, edgecolor="#2d3748")
        ax2.axhline(budama_esigi, color="#d69e2e", linestyle="--", linewidth=2, label=f"Budama Eşiği ({budama_esigi:.2f})")

        ax2.set_title("2. Conv Katmanı Filtre L1 Norm Skorları (Kırmızı: Budanacak)", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.set_xlabel("Filtre / Kanal İndeksi", fontsize=10)
        ax2.set_ylabel("L1 Norm Değeri (||W||_1)", fontsize=10)
        ax2.legend(loc="upper left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 3: Budama Oranı vs Parametre Azalması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        bars3 = ax3.bar(oranlar, parametreler, color=["#4a5568", "#3182ce", "#805ad5"], width=0.45, edgecolor="#2d3748")
        ax3.set_title("3. Budama Oranına Göre Parametre Tasarrufu", fontsize=12, fontweight="bold", color="#2d3748")
        ax3.set_ylabel("Toplam Parametre Sayısı", fontsize=10)
        ax3.set_ylim(0, max(parametreler) * 1.25)

        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval + (max(parametreler)*0.03), f"{yval:,}", ha="center", va="bottom", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Doğruluk Toparlanması (Fine-Tuning Öncesi vs Sonrası)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        x_idx = np.arange(len(oranlar))
        w = 0.35

        ax4.bar(x_idx - w/2, dogruluk_oncesi, width=w, label="Budama Hemen Sonrası", color="#e53e3e", alpha=0.85)
        ax4.bar(x_idx + w/2, dogruluk_sonrasi, width=w, label="Fine-Tuning Sonrası", color="#38a169", alpha=0.85)

        ax4.set_title("4. Doğruluk (Val Top-1 %) ve İnce Ayar Toparlanması", fontsize=12, fontweight="bold", color="#22543d")
        ax4.set_xlabel("Budama Oranı", fontsize=10)
        ax4.set_ylabel("Doğruluk (%)", fontsize=10)
        ax4.set_xticks(x_idx)
        ax4.set_xticklabels(oranlar)
        ax4.set_ylim(0, 115)
        ax4.legend(loc="lower left", frameon=True)

        for i in range(len(oranlar)):
            ax4.text(x_idx[i] - w/2, dogruluk_oncesi[i] + 2, f"%{dogruluk_oncesi[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
            ax4.text(x_idx[i] + w/2, dogruluk_sonrasi[i] + 2, f"%{dogruluk_sonrasi[i]:.1f}", ha="center", fontsize=9, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 5: Çıkarım Gecikmesi (Latency ms) & Hızlanma
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(oranlar, gecikmeler, "r-o", linewidth=2.5, markersize=8, label="Gecikme (ms)")
        ax5.set_title("5. Fiziksel Çıkarım Gecikmesi Azalması (ms)", fontsize=12, fontweight="bold", color="#c53030")
        ax5.set_xlabel("Budama Oranı", fontsize=10)
        ax5.set_ylabel("Gecikme (ms / batch)", fontsize=10)
        ax5.set_ylim(0, max(gecikmeler) * 1.3)

        for i, txt in enumerate(gecikmeler):
            ax5.annotate(f"{txt:.2f} ms", (oranlar[i], gecikmeler[i] + (max(gecikmeler)*0.05)), fontweight="bold", ha="center")
        ax5.legend(loc="upper right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       STRUCTURED PRUNING (YAPISAL BUDAMA) SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Özel kütüphane gerektirmeksizin tüm donanımlarda anında hızlanma.\n"
            "  • Tensör boyutlarını ve bellek ayak izini doğrudan küçültür.\n"
            "  • FLOPs ve güç tüketimini doğrusal olarak azaltır.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Çok yüksek budama oranlarında (> %60) dramatik doğruluk kaybı.\n"
            "  • Katman dikişi (Layer Stitching) kodlama karmaşıklığı gerektirir.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • INT8 Kuantizasyon ve Knowledge Distillation ile birleştirilebilir.\n"
            "  • Düşük güçlü Edge AI çiplerinde yüksek FPS çalıştırma.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Fine-tuning yapılmazsa kritik temsiller kaybolabilir."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Structured Pruning SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
