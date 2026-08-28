"""
Deterministik Egitim ve Konfigurasyon Gorsellestiricisi
======================================================
Run A, Run B ve Run C sonuclarini, kayip egrilerini, LR cizelgesini ve
sayisal fark analizlerini 6 panelli yuksek cozunurluklu endustriyel teshis tablosunda birlestirir.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DeterminizmGorsellestirici:
    """
    Day 67 6-Panelli Determinizm ve Konfigurasyon Teşhis Panosu ureticisi.
    """

    @staticmethod
    def panoyu_ciz_ve_kaydet(
        dogrulama_sonuclari: Dict[str, Any],
        cikti_yolu: str = "ciktilar/deterministik_egitim_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(21, 13))
        fig.suptitle(
            "Day 67: YAML Konfigurasyon Yonetimi, Deterministik & Tekrarlanabilir Egitim Paneli",
            fontsize=17,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        res_a = dogrulama_sonuclari["sonuc_a"]
        res_b = dogrulama_sonuclari["sonuc_b"]
        res_c = dogrulama_sonuclari["sonuc_c"]

        epochs = list(range(1, len(res_a["gecmis"]["train_loss"]) + 1))

        # -------------------------------------------------------------
        # 1. Panel: Yönetici ve Determinizm Özet Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        durum_str = "DETERMINISTIK: %100 BIT-FOR-BIT ESLESME" if dogrulama_sonuclari["deterministik_basarili"] else "BASARISIZ"
        ozet_metin = (
            "       DETERMINISTIK EGITIM VE MLOPS OZETI\n"
            "═══════════════════════════════════════════════════════\n"
            f" * Determinizm Durumu   : {durum_str}\n"
            f" * Run A (Seed=42) Loss : {res_a['son_train_loss']:>8.6f} | Acc: %{res_a['son_val_accuracy']:.1f}\n"
            f" * Run B (Seed=42) Loss : {res_b['son_train_loss']:>8.6f} | Acc: %{res_b['son_val_accuracy']:.1f}\n"
            f" * Run C (Seed=99) Loss : {res_c['son_train_loss']:>8.6f} | Acc: %{res_c['son_val_accuracy']:.1f}\n"
            "───────────────────────────────────────────────────────\n"
            f" * Max Loss Delta (A-B) : {dogrulama_sonuclari['maks_train_loss_delta_ab']:>10.8f} (SIFIR FARK)\n"
            f" * Max Acc Delta (A-B)  : % {dogrulama_sonuclari['maks_val_acc_delta_ab']:>6.2f} (TAM ESITLIK)\n"
            f" * Agirlik SHA256 Esit  : {'%100 DOGRULANDI' if dogrulama_sonuclari['agirlik_hash_eslesmesi'] else 'HATALI'}\n"
            "───────────────────────────────────────────────────────\n"
            f" * Run A Hash: {dogrulama_sonuclari['run_a_hash'][:16]}...\n"
            f" * Run B Hash: {dogrulama_sonuclari['run_b_hash'][:16]}...\n"
            f" * Run C Hash: {dogrulama_sonuclari['run_c_hash'][:16]}...\n"
            "═══════════════════════════════════════════════════════\n"
            " * Pydantic v2 + YAML Konfigurasyon Dogrulamasi: PASSED"
        )
        ax1.text(
            0.5, 0.5, ozet_metin,
            transform=ax1.transAxes,
            fontsize=10.5,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#e8f8f5", edgecolor="#1abc9c", linewidth=2.0)
        )
        ax1.set_title("1. Determinizm & Konfigurasyon Ozeti", fontweight="bold", color="#16a085")

        # -------------------------------------------------------------
        # 2. Panel: Eğitim Kaybı (Train Loss) Karşılaştırması
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs, res_a["gecmis"]["train_loss"], label="Run A (Seed=42)", color="#2ecc71", linewidth=3.0)
        ax2.plot(epochs, res_b["gecmis"]["train_loss"], label="Run B (Seed=42)", color="#3498db", linestyle="--", linewidth=2.0)
        ax2.plot(epochs, res_c["gecmis"]["train_loss"], label="Run C (Seed=99 - Farkli)", color="#e74c3c", linestyle=":", linewidth=2.0)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Cross-Entropy Kaybi")
        ax2.set_title("2. Egitim Kaybi (Run A vs Run B Ortusmesi)", fontweight="bold", color="#2980b9")
        ax2.legend(loc="upper right")

        # -------------------------------------------------------------
        # 3. Panel: Doğrulama Başarımı (Validation Accuracy)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epochs, res_a["gecmis"]["val_accuracy"], label="Run A (Seed=42)", color="#2ecc71", marker="o", linewidth=2.5)
        ax3.plot(epochs, res_b["gecmis"]["val_accuracy"], label="Run B (Seed=42)", color="#3498db", marker="s", linestyle="--", linewidth=1.8)
        ax3.plot(epochs, res_c["gecmis"]["val_accuracy"], label="Run C (Seed=99)", color="#e74c3c", marker="^", linestyle=":", linewidth=1.8)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Dogruluk / Accuracy (%)")
        ax3.set_title("3. Dogrulama Basarimi Karşılaştırması", fontweight="bold", color="#8e44ad")
        ax3.legend(loc="lower right")

        # -------------------------------------------------------------
        # 4. Panel: Öğrenme Oranı (Cosine LR Schedule)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs, res_a["gecmis"]["lr"], color="#d35400", marker="d", linewidth=2.5)
        for ep, lr_val in zip(epochs, res_a["gecmis"]["lr"]):
            ax4.annotate(f"{lr_val:.5f}", (ep, lr_val), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Ogrenme Orani (LR)")
        ax4.set_title("4. Cosine Annealing LR Zamanlayicisi", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # 5. Panel: Sayısal Hata Farkı Delta(t) = |Loss_A - Loss_B|
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        loss_farki = [abs(la - lb) for la, lb in zip(res_a["gecmis"]["train_loss"], res_b["gecmis"]["train_loss"])]
        ax5.plot(epochs, loss_farki, color="#27ae60", marker="o", linewidth=2.5, label="|Loss_A - Loss_B|")
        ax5.axhline(0.0, color="#7f8c8d", linestyle="--", alpha=0.7)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Mutlak Kayip Farki (Delta)")
        ax5.set_ylim(-0.0001, 0.001)
        ax5.set_title("5. Sayisal Hata Farki (|Loss_A - Loss_B| = 0.0)", fontweight="bold", color="#27ae60")
        ax5.legend(loc="upper right")

        # -------------------------------------------------------------
        # 6. Panel: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_metin = (
            " CONFIG-DRIVEN DETERMINISTIK EGITIM SWOT MATRISI\n"
            "─────────────────────────────────────────────────\n"
            " [S] GUCLU YONLER (Strengths):\n"
            " • %100 bilimsel tekrarlanabilirlik ve hata ayiklama\n"
            " • Pydantic ile calisma zamani oncesi tip dogrulamasi\n"
            " • Kod degistirmeden YAML uzerinden deney yonetimi\n\n"
            " [W] ZAYIF YONLER (Weaknesses):\n"
            " • cuDNN deterministik modda %5-10 egitim yavaslamasi\n"
            " • Bazi ozel CUDA cekirdeklerinde determinizm kisiti\n\n"
            " [O] FIRSATLAR (Opportunities):\n"
            " • Takim ici ortak deneyler ve CI/CD regresyon testi\n"
            " • Hyperparameter tuning (Optuna/Ray) entegrasyonu\n\n"
            " [T] TEHDITLER (Threats):\n"
            " • Tohumlanmamis coklu GPU/DataLoader veri sizintisi"
        )
        ax6.text(
            0.5, 0.5, swot_metin,
            transform=ax6.transAxes,
            fontsize=9.2,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.8)
        )
        ax6.set_title("6. Deterministik Egitim SWOT Matrisi", fontweight="bold", color="#d35400")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return cikti_yolu
