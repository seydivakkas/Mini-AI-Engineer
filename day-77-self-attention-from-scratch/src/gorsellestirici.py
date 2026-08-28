"""
Self-Attention Teşhis ve Analiz Görselleştiricisi
------------------------------------------------
6 panelli yüksek çözünürlüklü Multi-Head Self-Attention görselleştirme panosu.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


class DikkatGorsellestirici:
    """
    Çok kafalı dikkat haritalarını, ölçekleme etkisini ve baş çeşitliliğini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_teshis_paneli(
        self,
        dikkat_haritalari: torch.Tensor,
        olcek_analizi: Dict[str, Any],
        entropiler: np.ndarray,
        mesafeler: np.ndarray,
        bas_cesitliligi: float,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Self-Attention teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 77: Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention (MHSA) Analiz Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: MHSA Matematiksel Akış Şeması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        akisk_metni = (
            "          MULTI-HEAD SELF-ATTENTION (MHSA) AKIŞI\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. GİRDİ TEMSİLİ: X ∈ ℝ^(B × N × D_model)\n\n"
            "  2. DOĞRUSAL PROJEKSİYONLAR:\n"
            "     • Q = X · W_Q ,  K = X · W_K ,  V = X · W_V\n\n"
            "  3. KAFALARA BÖLME (H adet Baş, d_k = D_model / H):\n"
            "     • Q_h, K_h, V_h ∈ ℝ^(B × H × N × d_k)\n\n"
            "  4. ÖLÇEKLİ NOKTA ÇARPIM DİKKATİ:\n"
            "     • Skor = (Q_h · K_h^T) / √d_k\n"
            "     • A_h = Softmax(Skor + Mask)\n"
            "     • Head_h = A_h · V_h\n\n"
            "  5. BİRLEŞTİRME & ÇIKIŞ PROJEKSİYONU:\n"
            "     • MultiHead = Concat(Head_1, ..., Head_H) · W_O"
        )
        ax1.text(
            0.5, 0.5, akisk_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. MHSA Matematiksel Hesaplama Akışı", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: 4 Başın Dikkat Haritaları (Heatmap Grid)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        # İlk batch örneğinin 4 başını birleştir: (4, N, N)
        A_sample = dikkat_haritalari[0].detach().cpu().numpy()
        H, N, _ = A_sample.shape
        
        # 2x2 grid olarak tek bir subplot içinde görselleştir
        grid_map = np.zeros((2 * N + 2, 2 * N + 2))
        grid_map[:N, :N] = A_sample[0]
        grid_map[:N, N+2:] = A_sample[1]
        grid_map[N+2:, :N] = A_sample[2]
        grid_map[N+2:, N+2:] = A_sample[3]

        sns.heatmap(grid_map, ax=ax2, cmap="viridis", cbar=True, square=True)
        ax2.set_title(f"2. Multi-Head Dikkat Isı Haritası (Head 1-4, N={N})", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.set_xticks([N//2, N + 2 + N//2])
        ax2.set_xticklabels(["Head 1 / Head 3", "Head 2 / Head 4"], fontsize=9)
        ax2.set_yticks([N//2, N + 2 + N//2])
        ax2.set_yticklabels(["Üst Başlar", "Alt Başlar"], fontsize=9)

        # -------------------------------------------------------------
        # PANEL 3: sqrt(d_k) Ölçeklemenin Softmax Doygunluğuna Etkisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        skor_unscaled = olcek_analizi["skor_olceksiz_ornek"]
        skor_scaled = olcek_analizi["skor_olcekli_ornek"]
        
        x_indices = np.arange(len(skor_unscaled))
        p_unscaled = np.exp(skor_unscaled) / np.sum(np.exp(skor_unscaled))
        p_scaled = np.exp(skor_scaled) / np.sum(np.exp(skor_scaled))

        ax3.plot(x_indices, p_unscaled, "r-o", linewidth=2, label=f"Ölçeksiz (Entropi: {olcek_analizi['olceksiz_entropi']:.2f})")
        ax3.plot(x_indices, p_scaled, "b-s", linewidth=2, label=f"1/√d_k Ölçekli (Entropi: {olcek_analizi['olcekli_entropi']:.2f})")

        ax3.set_title("3. √d_k Ölçeklemenin Softmax Doygunluğuna Etkisi", fontsize=12, fontweight="bold", color="#c53030")
        ax3.set_xlabel("Hedef Token İndeksi (Key Token)", fontsize=10)
        ax3.set_ylabel("Dikkat Olasılığı P(A_j)", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Token Başına Dikkat Yayılımı ve Alıcı Alan
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        bas_isimleri = [f"Baş {h+1}" for h in range(H)]
        x_pos = np.arange(H)
        
        ax4.bar(x_pos, mesafeler, color="#38a169", width=0.5, alpha=0.85)
        for i, val in enumerate(mesafeler):
            ax4.text(i, val + 0.1, f"{val:.2f} px", ha="center", fontsize=9, fontweight="bold")

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(bas_isimleri, fontsize=10)
        ax4.set_ylabel("Ortalama Dikkat Mesafesi (|i - j|)", fontsize=10)
        ax4.set_title("4. Başların Uzamsal Alıcı Alanı (Lokal vs Global)", fontsize=12, fontweight="bold", color="#276749")
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Başlar Arası Entropi ve Çeşitlilik
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        colors = ["#3182ce", "#805ad5", "#dd6b20", "#e53e3e"]
        bars = ax5.bar(bas_isimleri, entropiler, color=colors[:H], width=0.5, alpha=0.85)
        for bar, val in zip(bars, entropiler):
            ax5.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")

        ax5.set_ylabel("Shannon Entropisi (Bit)", fontsize=10)
        ax5.set_title(f"5. Baş Entropisi (Çeşitlilik: {bas_cesitliligi:.3f})", fontsize=12, fontweight="bold", color="#4a5568")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "          MULTI-HEAD SELF-ATTENTION SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • O(1) yol uzunluğu ile küresel bağlam yakalama (Global Context).\n"
            "  • Çoklu başlar sayesinde eşzamanlı farklı ilişkileri öğrenme.\n"
            "  • Paralelleştirilebilir matris çarpımları (GPU için mükemmel).\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Dizi uzunluğuna göre karesel O(N^2) bellek ve zaman karmaşıklığı.\n"
            "  • Evrişim (CNN) gibi doğal konumsal tümevarım yanlılığı (Inductive Bias) yok.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • FlashAttention ile GPU SRAM üzerinde O(N^2) bellek duvarını aşma.\n"
            "  • Vision Transformers (ViT) ile bilgisayarlı görüde SOTA doğruluk.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Küçük veri kümelerinde yetersiz ön eğitimle aşırı uydurma (Overfitting)."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. MHSA Mimari SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
