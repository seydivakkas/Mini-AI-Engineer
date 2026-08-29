"""
LLM Filigran ve Tespit Teşhis Panosu Görselleştirici Modülü (Day 118).
6 panelli Z-skoru dağılımı, yeşil token oranı, delta duyarlılığı, paraphrase dayanıklılığı ve kriptografik akış şeması.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class FiligranGorsellestirici:
    """LLM filigranlama ve istatistiksel tespit analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/filigran_tespit_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Kirchenbauer LLM Kriptografik Filigranlama (Watermarking) ve Z-Skoru Tespit Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Z-Skoru Dağılımı ve Eşik Çizgisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.hist(rapor["filigransiz_z"], bins=10, alpha=0.6, color="#e74a3b", label="Filigransız / İnsan", edgecolor="black")
        ax1.hist(rapor["filigranli_z"], bins=10, alpha=0.7, color="#1cc88a", label="Filigranlı (AI Model)", edgecolor="black")
        ax1.axvline(4.0, color="black", linestyle="--", lw=2.5, label="Karar Eşiği (Z=4.0)")

        ax1.set_title("1. Z-Skoru Dağılımı (Filigranlı vs Filigransız)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Z-Skoru")
        ax1.set_ylabel("Örnek Sayısı")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Yeşil Token Oranı Karşılaştırması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        gruplar = ["Filigransız\n(Beklenen: %50)", "Filigranlı AI\n(Gözlenen)"]
        oranlar = [rapor["filigransiz_yesil_oran"] * 100.0, rapor["filigranli_yesil_oran"] * 100.0]
        renkler2 = ["#6c757d", "#28a745"]

        barlar2 = ax2.bar(gruplar, oranlar, color=renkler2, edgecolor="black", width=0.5)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax2.axhline(50.0, color="red", linestyle=":", lw=2, label="Rastgele Dağılım (%50)")
        ax2.set_title("2. Yeşil Liste Token Oranı (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Yeşil Token Yüzdesi (%)")
        ax2.set_ylim(0, 110)
        ax2.legend(loc="lower right")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Delta Duyarlılığı (Watermark Strength)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(rapor["delta_degerleri"], rapor["delta_z_skorlari"], marker="o", lw=3.0, color="#4e73df", markersize=8)
        for d, z in zip(rapor["delta_degerleri"], rapor["delta_z_skorlari"]):
            ax3.annotate(f"Z={z:.1f}", (d, z + 0.5), ha="center", fontweight="bold", fontsize=9)

        ax3.axhline(4.0, color="red", linestyle="--", lw=1.5, label="Tespit Eşiği (Z=4.0)")
        ax3.set_title("3. Delta Logit Yanlılığına Göre Z-Skoru Artışı", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Delta (δ) Yanlılık Değeri")
        ax3.set_ylabel("Ortalama Z-Skoru")
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Paraphrase / Düzenleme Saldırısı Dayanıklılığı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        edit_yuzdeleri = [f"%{int(e * 100)}" for e in rapor["edit_oranlari"]]
        ax4.plot(edit_yuzdeleri, rapor["paraphrase_z"], marker="s", lw=3.0, color="#e67e22", markersize=8)
        for e, z in zip(edit_yuzdeleri, rapor["paraphrase_z"]):
            ax4.annotate(f"Z={z:.1f}", (e, z + 0.4), ha="center", fontweight="bold", fontsize=9)

        ax4.axhline(4.0, color="red", linestyle="--", lw=1.5, label="Tespit Sınırı (Z=4.0)")
        ax4.set_title("4. Metin Değiştirme (Paraphrase) Dayanıklılığı", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Rastgele Değiştirilen Token Oranı")
        ax4.set_ylabel("Kalan Z-Skoru")
        ax4.legend(loc="upper right")
        ax4.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: Kirchenbauer Filigran Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Kirchenbauer Filigranlama Akış Şeması", fontsize=12, fontweight="bold", pad=10)

        akis_semasi = (
            "====================================================\n"
            "       KIRCHENBAUER GREEN/RED WATERMARK FLOW        \n"
            "====================================================\n"
            "1. ÖNCEKİ TOKEN (x_{t-1}) & GİZLİ ANAHTAR (Key):\n"
            "   • Hash = SHA256(x_{t-1} || Key) % 2^32\n\n"
            "2. SÖZLÜK BÖLÜMLEME (Vocabulary Partitioning):\n"
            "   • Yeşil Liste (G): Kelimelerin %50'si (gamma = 0.5)\n"
            "   • Kırmızı Liste (R): Kalan %50 kelime\n\n"
            "3. LOGIT YANLILIĞI (Logit Biasing):\n"
            "   • logits[G] = logits[G] + delta (örnek: +2.5)\n"
            "   • Softmax & Örnekleme -> Yeşil Token Üretilir!\n\n"
            "4. Z-SKORU İLE DOĞRULAMA (Detection):\n"
            "   • Z = (|G| - 0.5 * T) / sqrt(0.25 * T)\n"
            "   • Z >= 4.0 ise p < 0.00003 -> KESİN AI METNİ!\n"
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
        # PANEL 6: LLM Kriptografik Filigran Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. LLM Kriptografik Filigran Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "       LLM WATERMARK VERIFICATION CERTIFICATE       \n"
            "====================================================\n"
            f"• Filigranlı Ortalama Z-Skoru : Z = {rapor['filigranli_ort_z']:.2f}\n"
            f"• Filigransız Z-Skoru         : Z = {rapor['filigransiz_ort_z']:.2f}\n"
            f"• Doğru Tespit Oranı (TPR)    : %{rapor['tpr_dogru_tespit_orani']:.1f}\n"
            f"• Yanlış Alarm Oranı (FPR)    : %{rapor['fpr_yanlis_alarm_orani']:.1f}\n"
            "• Algoritmik Standart         : Kirchenbauer et al.\n"
            "                                (ICML 2023 En İyi Makale)\n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Matematiksel Olarak Kanıtlanabilir AI İmzası\n"
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
        print(f"  ✓ Filigran Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
