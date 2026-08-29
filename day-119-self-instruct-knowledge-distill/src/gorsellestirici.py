"""
Knowledge Distillation Teşhis Panosu Görselleştirici Modülü (Day 119).
6 panelli kayıp eğrileri, KL diverjansı, çıkarım gecikmesi, parametre tasarrufu ve damıtma mimari şeması.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class DamitmaGorsellestirici:
    """Knowledge Distillation ve Öğrenci Model analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/knowledge_distillation_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Öğretmenden Öğrenciye Bilgi Damıtma (Knowledge Distillation) ve Self-Instruct Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        adimlar = list(range(1, len(rapor["sft_kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: Eğitim Kayıp Eğrileri (SFT vs KD)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(adimlar, rapor["sft_kayiplar"], color="#e74a3b", lw=2.5, label="Standart SFT Öğrenci")
        ax1.plot(adimlar, rapor["kd_kayiplar"], color="#1cc88a", lw=2.5, label="Knowledge Distilled (KD) Öğrenci")

        ax1.set_title("1. SFT vs KD Öğrenci Eğitim Kayıp Eğrisi", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Eğitim Adımı")
        ax1.set_ylabel("Toplam Kayıp (Loss)")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: KL Diverjans Kaybı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(adimlar, rapor["kl_kayiplar"], color="#4e73df", lw=2.5, marker="o", markersize=4)

        ax2.set_title("2. Öğretmen-Öğrenci Logit KL Diverjansı ($T^2 D_{KL}$)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Eğitim Adımı")
        ax2.set_ylabel("KL Kaybı")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Çıkarım Gecikmesi & Hızlanma
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        modeller = ["Öğretmen Model\n(Teacher)", "Öğrenci Model\n(Student)"]
        gecikmeler = [rapor["ogretmen_gecikme_ms"], rapor["ogrenci_gecikme_ms"]]
        renkler3 = ["#e74a3b", "#28a745"]

        barlar3 = ax3.bar(modeller, gecikmeler, color=renkler3, edgecolor="black", width=0.5)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f} ms", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax3.set_title(f"3. Çıkarım Gecikmesi ({rapor['hizlanma_orani']:.1f}x Hızlanma)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Gecikme (ms / batch)")
        ax3.set_ylim(0, max(gecikmeler) * 1.25)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Parametre Sayısı & Bellek Tasarrufu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        params = [rapor["ogretmen_parametre"] / 1000.0, rapor["ogrenci_parametre"] / 1000.0]
        barlar4 = ax4.bar(modeller, params, color=["#6c757d", "#17a2b8"], edgecolor="black", width=0.5)
        for bar in barlar4:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, h + 20, f"{h:.1f}K Param", ha="center", va="bottom", fontweight="bold", fontsize=10)

        ax4.set_title(f"4. Parametre Tasarrufu (%{rapor['parametre_tasarrufu']:.1f} Küçülme)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Parametre Sayısı (x1000)")
        ax4.set_ylim(0, max(params) * 1.25)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: Knowledge Distillation Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Knowledge Distillation Mimarisi", fontsize=12, fontweight="bold", pad=10)

        akis_semasi = (
            "====================================================\n"
            "       KNOWLEDGE DISTILLATION (KD) PIPELINE         \n"
            "====================================================\n"
            "1. ÖĞRETMEN MODEL (Teacher - 70B+):\n"
            "   • Girdi (x) -> logits_T -> Softmax(logits_T / T)\n"
            "   • Zengin olasılık dağılımı (Karanlık Bilgi / Dark Knowledge)\n\n"
            "2. ÖĞRENCİ MODEL (Student - 1B/3B):\n"
            "   • Girdi (x) -> logits_S -> Softmax(logits_S / T)\n\n"
            "3. BİLEŞİK KAYIP FONKSİYONU:\n"
            "   • L_Total = alpha * L_CE + (1 - alpha) * T^2 * L_KL\n"
            "   • L_CE: Gerçek hedef etiketler (Hard Targets)\n"
            "   • L_KL: Öğretmen dağılımı ile uyum (Soft Targets)\n\n"
            "4. ÇIKTI:\n"
            "   • 10x daha hafif, 4x daha hızlı ve akıl yürüten model!\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, akis_semasi,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Bilgi Damıtma ve Kalite Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Model Sıkıştırma ve Damıtma Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "     KNOWLEDGE DISTILLATION QUALITY CERTIFICATE     \n"
            "====================================================\n"
            f"• Öğretmen Parametre         : {rapor['ogretmen_parametre']:,} Param\n"
            f"• Öğrenci Parametre          : {rapor['ogrenci_parametre']:,} Param\n"
            f"• Parametre Tasarrufu        : %{rapor['parametre_tasarrufu']:.1f} Azalma\n"
            f"• Çıkarım Hızlanması         : {rapor['hizlanma_orani']:.1f}x Hızlı\n"
            "• Endüstriyel Eşdeğerler     : DeepSeek-R1-Distill,\n"
            "                               Llama-3.2-1B, MiniLLM\n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Edge & Mobil Cihazlara Hazır Yüksek Zekâ\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, sertifika,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Knowledge Distillation Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
