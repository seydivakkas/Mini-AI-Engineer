"""
KTO Tercih Hizalaması ve Beklenti Teorisi Teşhis Panosu Görselleştirici Modülü (Day 111).
6 panelli KTO kayıp eğrisi, örtük ödül ayrışması, doğruluk, Prospect Theory eğrisi ve matematik kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class KTOGorsellestirici:
    """KTO analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, List[float]],
        kayit_yolu: str = "ciktilar/kto_alignment_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Kahneman-Tversky Optimization (KTO): Eşleştirilmemiş İkili Geri Bildirimlerle LLM Hizalama Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        epoklar = list(range(1, len(egitim_raporu["toplam_kayiplar"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: KTO Kayıp Eğrileri (Total, Desirable, Undesirable)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, egitim_raporu["toplam_kayiplar"], color="#4e73df", lw=3.0, marker="o", label="Toplam KTO Kaybı")
        ax1.plot(epoklar, egitim_raporu["kayiplar_d"], color="#1cc88a", lw=2.0, linestyle="--", label="Desirable Kaybı (L_D)")
        ax1.plot(epoklar, egitim_raporu["kayiplar_u"], color="#e74a3b", lw=2.0, linestyle=":", label="Undesirable Kaybı (L_U)")
        ax1.set_title("1. KTO Asimetrik Kayıp Eğrileri (Epoklar Boyunca)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epok")
        ax1.set_ylabel("Kayıp Değeri")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 2: İkili Doğruluk Oranı (% Binary Accuracy)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epoklar, egitim_raporu["dogruluklar"], color="#28a745", lw=3.0, marker="s", label="KTO Doğruluğu (%)")
        ax2.axhline(50.0, color="gray", linestyle=":", label="Rastgele Eşik (%50)")
        ax2.axhline(95.0, color="#e74a3b", linestyle="--", label="Hedef Başarı (%95)")
        ax2.set_title("2. İkili Tercih Doğruluğu (Binary Alignment Accuracy)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epok")
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_ylim(40, 105)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="lower right")

        # -------------------------------------------------------------
        # PANEL 3: Örtük Ödüllerin Ayrışması (r_D vs r_U & Delta r)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epoklar, egitim_raporu["r_d_ort"], color="#1cc88a", lw=2.5, marker="^", label=r"$\hat{r}_D$ (Beğenilen Ödül)")
        ax3.plot(epoklar, egitim_raporu["r_u_ort"], color="#e74a3b", lw=2.5, marker="v", label=r"$\hat{r}_U$ (Beğenilmeyen Ödül)")
        ax3.plot(epoklar, egitim_raporu["marjinler"], color="#6f42c1", lw=2.0, linestyle="--", label=r"$\Delta \hat{r} = \hat{r}_D - \hat{r}_U$")
        ax3.set_title("3. Örtük Ödüllerin ve Marjinin Ayrışması", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Epok")
        ax3.set_ylabel("Örtük Ödül (Implicit Reward)")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend(loc="center left")

        # -------------------------------------------------------------
        # PANEL 4: Beklenti Teorisi (Prospect Theory) Değer Fonksiyonu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        x_vals = np.linspace(-4, 4, 200)
        # S-Curve: Kazançlar konkav (1.0), Kayıplar konveks ve dik (1.33)
        v_vals = np.where(x_vals >= 0, 1.0 * (1.0 - 1.0 / (1.0 + np.exp(x_vals))), -1.33 * (1.0 - 1.0 / (1.0 + np.exp(-x_vals))))

        ax4.plot(x_vals, v_vals, color="#e74a3b", lw=3.0, label="Kahneman-Tversky S-Eğrisi")
        ax4.axvline(0, color="black", linestyle="--", alpha=0.5, label="Referans Noktası (z_ref)")
        ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax4.fill_between(x_vals[x_vals >= 0], 0, v_vals[x_vals >= 0], color="#1cc88a", alpha=0.2, label="Kazanç Bölgesi (lambda_D=1.0)")
        ax4.fill_between(x_vals[x_vals < 0], 0, v_vals[x_vals < 0], color="#e74a3b", alpha=0.2, label="Kayıp Bölgesi (lambda_U=1.33 Dik!)")

        ax4.set_title("4. Beklenti Teorisi: Kayıp Kaçınması (Loss Aversion: λ_U > λ_D)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Ödül Sapması (r - z_ref)")
        ax4.set_ylabel("Psikolojik Değer / Ceza v(x)")
        ax4.grid(True, linestyle="--", alpha=0.7)
        ax4.legend(loc="lower right", fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 5: KTO Matematik Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. KTO Matematik ve Formül Kartı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] Kahneman-Tversky Optimization (KTO):\n"
            "--------------------------------------------------\n"
            "1. Örtük Ödül (Implicit Reward):\n"
            "   r(x,y) = beta * log( pi_theta(y|x) / pi_ref(y|x) )\n\n"
            "2. Referans Noktası (z_ref):\n"
            "   z_ref = E_D [ r(x,y) ] (Beklenen Ödül Referansı)\n\n"
            "3. Asimetrik Beklenti Kaybı (Prospect Loss):\n"
            "   L_D = lambda_D * ( 1 - sigma( r - z_ref ) )\n"
            "   L_U = lambda_U * ( 1 - sigma( z_ref - r ) )\n\n"
            "-> Çiftli veri (y_w vs y_l) gerekmez, tekil veri yeter!"
        )

        ax5.text(
            0.05, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & KTO Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & KTO Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "             KTO ALIGNMENT CERTIFICATE              \n"
            "====================================================\n"
            "• Soru: Gerçek dünyada neden KTO tercih edilir?     \n"
            "• Cevap: Kullanıcılar iki cevabı yarıştırmaz, sadece \n"
            "         beğendim/beğenmedim (thumbs up/down) der.   \n"
            "         KTO bu tekil verileri doğrudan eğitir!     \n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Contextual AI & Davranışsal Hizalama!   \n"
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
        print(f"  ✓ KTO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
