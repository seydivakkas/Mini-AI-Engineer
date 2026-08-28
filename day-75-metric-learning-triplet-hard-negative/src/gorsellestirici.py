"""
Triplet Metrik Öğrenimi Görselleştiricisi
-----------------------------------------
6 panelli yüksek çözünürlüklü teşhis ve mesafe dağılım panosu üreten modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TripletGorsellestirici:
    """
    Triplet öğrenim dinamiklerini, madencilik oranlarını ve manifold ayrışmasını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        gecmis: Dict[str, List[float]],
        gomulmeler_2d: np.ndarray,
        etiketler: np.ndarray,
        marjin: float,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli Triplet teşhis panosunu oluşturur ve diske kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 75: Triplet Margin Loss & Hard/Semi-Hard Negative Mining Teşhis Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        epochs = gecmis["epoch"]

        # -------------------------------------------------------------
        # PANEL 1: Triplet Madencilik Geometrisi Şeması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_xlim(-1.5, 2.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect("equal")
        
        # Çemberler
        cember_pos = plt.Circle((0, 0), 0.7, color="#3182ce", fill=False, linestyle="--", linewidth=1.5, label="d(a, p)")
        cember_marjin = plt.Circle((0, 0), 0.7 + marjin, color="#e53e3e", fill=False, linestyle="-.", linewidth=2, label="d(a, p) + α (Marjin)")
        ax1.add_patch(cember_pos)
        ax1.add_patch(cember_marjin)

        # Noktalar
        ax1.scatter([0], [0], color="#2b6cb0", s=200, zorder=5, label="Anchor (a)")
        ax1.scatter([0.7], [0], color="#38a169", s=180, zorder=5, label="Positive (p)")
        ax1.scatter([0.4], [0.3], color="#e53e3e", s=180, zorder=5, label="Hard Negative (d_an < d_ap)")
        ax1.scatter([0.85], [0.35], color="#dd6b20", s=180, zorder=5, label="Semi-Hard Negative (d_ap < d_an < d_ap+α)")
        ax1.scatter([1.6], [0.6], color="#718096", s=180, zorder=5, label="Easy Negative (d_an > d_ap+α)")

        ax1.set_title("1. Triplet Madencilik Geometrisi (Anchor, Positive, Negatif Tipleri)", fontsize=12, fontweight="bold", color="#1a365d")
        ax1.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 2: Triplet Margin Kayıp Trajektorisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs, gecmis["loss"], "o-", color="#e53e3e", linewidth=2.5, label="Triplet Kaybı")
        ax2.set_title(f"2. Triplet Margin Kayıp Eğrisi (Son: {gecmis['loss'][-1]:.4f})", fontsize=12, fontweight="bold", color="#9b2c2c")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Triplet Loss", fontsize=10)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, gecmis["aktif_oran"], "s--", color="#805ad5", linewidth=2, label="Aktif Triplet Oranı (%)")
        ax2_twin.set_ylabel("Aktif Triplet % (Loss > 0)", fontsize=10, color="#805ad5")
        
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: Metrik Temsil Uzayı (PCA İzdüşümü)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        benzersiz_siniflar = np.unique(etiketler)
        palette = sns.color_palette("bright", len(benzersiz_siniflar))
        for idx, c in enumerate(benzersiz_siniflar):
            mask = (etiketler == c)
            ax3.scatter(
                gomulmeler_2d[mask, 0], gomulmeler_2d[mask, 1],
                color=palette[idx], label=f"Sınıf {int(c)}",
                alpha=0.85, s=45
            )
        ax3.set_title("3. Öğrenilen Metrik Temsil Uzayı (Sınıf Kümeleri)", fontsize=12, fontweight="bold", color="#2c7a7b")
        ax3.set_xlabel("Embedding Boyut 1", fontsize=10)
        ax3.set_ylabel("Embedding Boyut 2", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Mesafe Gelişimi (d(a, p) vs d(a, n) ve Marjin)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs, gecmis["d_ap"], "o-", color="#38a169", linewidth=2.2, label="d(a, p) Pozitif Mesafe (Hedef -> 0)")
        ax4.plot(epochs, gecmis["d_an"], "s-", color="#e53e3e", linewidth=2.2, label="d(a, n) Negatif Mesafe (Hedef -> Geniş)")
        ax4.plot(epochs, gecmis["marjin"], "^-.", color="#3182ce", linewidth=2.5, label="Net Marjin (d_an - d_ap)")
        ax4.axhline(marjin, color="#dd6b20", linestyle=":", linewidth=2, label=f"Hedef Marjin α = {marjin}")
        
        son_mar = gecmis["marjin"][-1]
        ax4.set_title(f"4. Pozitif vs Negatif Mesafe Ayrışması (Marjin: {son_mar:.3f})", fontsize=12, fontweight="bold", color="#276749")
        ax4.set_xlabel("Epoch", fontsize=10)
        ax4.set_ylabel("Öklid Mesafesi", fontsize=10)
        ax4.legend(loc="center left", fontsize=8, framealpha=0.9)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Madencilik Dinamikleri (Zor vs Yarı-Zor vs Kolay)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        width = 0.55
        ep_arr = np.array(epochs)
        ax5.bar(ep_arr, gecmis["zor_oran"], width, label="Zor (Hard %)", color="#e53e3e", alpha=0.85)
        ax5.bar(ep_arr, gecmis["yari_zor_oran"], width, bottom=gecmis["zor_oran"], label="Yarı-Zor (Semi-Hard %)", color="#dd6b20", alpha=0.85)
        bottoms = np.array(gecmis["zor_oran"]) + np.array(gecmis["yari_zor_oran"])
        ax5.bar(ep_arr, gecmis["kolay_oran"], width, bottom=bottoms, label="Kolay (Easy %)", color="#718096", alpha=0.85)

        ax5.set_title("5. Triplet Dağılım Evrimi (Eğitim İlerledikçe Kolaylaşma)", fontsize=12, fontweight="bold", color="#4a5568")
        ax5.set_xlabel("Epoch", fontsize=10)
        ax5.set_ylabel("Oran (%)", fontsize=10)
        ax5.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "             TRIPLET METRIC LEARNING SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Yüz tanıma, Re-ID ve imza doğrulamada altın standarttır.\n"
            "  • Sabit sınıf sayısı sınırlaması yoktur (Open-Set Verification).\n"
            "  • Doğrudan Öklid mesafesi üzerinden net marjin garantisi sunar.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • O(N^3) olası triplet kombinasyonu; madencilik (mining) zorunludur.\n"
            "  • Yanlış Hard Mining stratejisi yerel minimumlara ve çökmeye yol açar.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Milyon ölçekli e-ticaret görsel arama ve biyometrik kimlik sistemleri.\n"
            "  • Few-shot / One-shot öğrenim görevlerinde üstün genelleme.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Gürültülü etiketlerde (label noise) Hard Negative madenciliği çöker."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Triplet Mimarisi SWOT Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
