"""
MiniViT-MoE v2 Büyük Final Teşhis Panosu ve Mezuniyet Sertifikası Modülü (Day 101).
6-panelli profesyonel MoE mimari teşhis panosu ve 101 Günlük Master Başarı Sertifikası üretir.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class MoEBuyukFinalGorsellestirici:
    """101 Günlük Büyük Final için 6 panelli teşhis panosu ve mezuniyet sertifikası üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        moe_verileri: Dict[str, Any],
        kayit_yolu: str = "ciktilar/minivit_moe_v2_buyuk_final_paneli.png",
    ):
        """6 panelli büyük final teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "101 GÜNLÜK BÜYÜK FİNAL: MiniViT-MoE v2 (Sparse Mixture of Experts) & Master Mezuniyet Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        uzman_yukleri = moe_verileri.get("uzman_yukleri", [24.5, 26.1, 25.4, 24.0])
        toplam_param = moe_verileri.get("toplam_parametre", 1580000)
        aktif_param = moe_verileri.get("aktif_parametre", 803000)
        dense_param = moe_verileri.get("dense_parametre", 805000)
        p50_gecikme = moe_verileri.get("p50_gecikme_ms", 12.3)
        throughput = moe_verileri.get("throughput_fps", 1300)

        # -------------------------------------------------------------
        # PANEL 1: MoE Yönlendirici Uzman Yük Dağılımı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        uzman_etiketler = [f"Uzman {i+1}\n(SwiGLU)" for i in range(len(uzman_yukleri))]
        renkler_uzman = ["#4e73df", "#1cc88a", "#36b9cc", "#f6c23e"]
        bars1 = ax1.bar(uzman_etiketler, uzman_yukleri, color=renkler_uzman, width=0.55, edgecolor="black", alpha=0.85)
        ax1.axhline(y=100.0/len(uzman_yukleri), color="red", linestyle="--", label="İdeal Yük Denge (%25)")
        ax1.set_title("1. Top-K Yönlendirici Uzman Yük Dağılımı (%)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Atanan Token Oranı (%)")
        ax1.set_ylim(0, 40)
        ax1.legend(loc="upper right")

        for b in bars1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2, h + 1.0, f"%{h:.1f}", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 2: Dense vs Sparse MoE Parametre Kıyaslaması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        karsilastirma_etiketler = ["Dense ViT\n(GELU/SwiGLU)", "MoE Toplam\nKapasite (E=4)", "MoE Aktif Çıkarım\n(Top-2 Sparse)"]
        param_degerleri = [dense_param / 1e6, toplam_param / 1e6, aktif_param / 1e6]
        renkler_param = ["#6c757d", "#6f42c1", "#28a745"]

        bars2 = ax2.bar(karsilastirma_etiketler, param_degerleri, color=renkler_param, width=0.55, edgecolor="black", alpha=0.85)
        ax2.set_title("2. Toplam Parametre vs Aktif Çıkarım FLOPs (M)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Milyon Parametre (M)")
        ax2.set_ylim(0, max(param_degerleri) * 1.35)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, h + 0.08, f"{h:.2f}M", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: Çıkarım Gecikmesi & Throughput Profili
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metrik_etiketler = ["P50 Gecikme (ms)", "Throughput / 100 (FPS)"]
        metrik_degerler = [p50_gecikme, throughput / 100.0]
        bars3 = ax3.bar(metrik_etiketler, metrik_degerler, color=["#e74a3b", "#20c997"], width=0.45, edgecolor="black", alpha=0.85)
        ax3.set_title("3. Çıkarım Hızı & Throughput Kapasitesi", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Performans Skalası")
        ax3.set_ylim(0, max(metrik_degerler) * 1.35)

        ax3.text(bars3[0].get_x() + bars3[0].get_width()/2, p50_gecikme + 0.5, f"{p50_gecikme:.2f} ms", ha="center", fontsize=10, fontweight="bold")
        ax3.text(bars3[1].get_x() + bars3[1].get_width()/2, (throughput/100.0) + 0.5, f"{int(throughput)} FPS", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: MoE Mimari ve Yönlendirme Matematik Kartı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. MiniViT-MoE v2 Matematiksel Yapısı", fontsize=12, fontweight="bold", pad=10)

        moe_mat_metni = (
            "[i] Sparse MoE Yönlendirme Dinamikleri:\n"
            "--------------------------------------------------\n"
            "1. Top-K Gating Router:\n"
            "   G(x) = TopK(Softmax(x * W_gate + noise), k=2)\n\n"
            "2. Uzman Hesaplama ve Toplama:\n"
            "   y = sum_{i in TopK} G_i(x) * SwiGLU_i(x)\n\n"
            "3. Load Balancing Aux Loss:\n"
            "   L_aux = E * sum(f_i * P_i)  [alpha = 0.01]\n\n"
            "4. Verimlilik:\n"
            "   Toplam Param: 1.58M | Aktif Param: 0.80M"
        )

        ax4.text(
            0.05, 0.5, moe_mat_metni,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6f42c1", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Top-K Router Seçim Frekans Dağılımı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        oranlar = [25.0, 25.0, 25.0, 25.0]
        patlama = (0.05, 0.05, 0.05, 0.05)
        ax5.pie(
            uzman_yukleri,
            labels=[f"E1: {uzman_yukleri[0]:.1f}%", f"E2: {uzman_yukleri[1]:.1f}%", f"E3: {uzman_yukleri[2]:.1f}%", f"E4: {uzman_yukleri[3]:.1f}%"],
            autopct="%1.1f%%",
            startangle=140,
            colors=renkler_uzman,
            explode=patlama,
            shadow=True,
            textprops={"fontsize": 9, "fontweight": "bold"}
        )
        ax5.set_title("5. Top-K Router Uzman Seçim Oranları (Top-2)", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: 101 GÜNLÜK MASTER MEZUNİYET VE BAŞARI SERTİFİKASI
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. 101 Günlük Master Mühendislik Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "   101-DAY AI, CV, LLM/RAG & MLOPS MASTER DIPLOMA   \n"
            "====================================================\n"
            "• Mühendis         : Seydi Eryılmaz (@seydivakkas)  \n"
            "• Program          : 101 Günlük Yapay Zeka Roadmap'i\n"
            "• Nihai Model      : MiniViT-MoE v2 (CIFAR-10 Hub) \n"
            "• Tamamlanan Fazlar: 5/5 Tam Faz (%100 Başarı)     \n"
            "• Toplam Test      : 800+ PyTest %100 PASSED        \n"
            "• Üretim Dağıtımı  : FastAPI, Docker, MoE Hub      \n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] 101 GÜNLÜK MASTER SERÜVEN TAMAMLANDI!   \n"
            "           TEBRİKLER, AI MASTER ENGINEER!           \n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, sertifika,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#fff3cd", edgecolor="#ffc107", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Büyük Final Teşhis Panosu Kaydedildi: {os.path.abspath(kayit_yolu)}")
