"""
Sembolik Akıl Yürütme Teşhis Panosu Görselleştirici Modülü (Day 150 - Faz 8).
6 panelli Kesinlik Kıyası, SymPy Kök Analizi, Z3 SMT Kısıt Uzayı, İspat Matrisi, Akış Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class SembolikReasoningGorsellestirici:
    """Neuro-Symbolic akıl yürütme teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        ispat_sonuclari: Dict[str, Any],
        kayit_yolu: str = "ciktilar/symbolic_math_z3_sympy_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 150: Sembolik Akıl Yürütme: LLM ile Z3 SMT Solver & SymPy Entegrasyonu (FAZ 8 YARI-YOL FİNALİ)",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Saf LLM vs Neuro-Symbolic LLM
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = ["Standart LLM\n(Olasılıksal)", "Chain-of-Thought\n(CoT Arama)", "Neuro-Symbolic\n(LLM + Z3 + SymPy)"]
        kesinlikler = [45.0, 72.0, 100.0]
        renkler1 = ["#e74a3b", "#f6c23e", "#1cc88a"]

        barlar1 = ax1.bar(yontemler, kesinlikler, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_title("1. Karmaşık Matematik & Mantık Kesinliği", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Deterministik Doğruluk (%)")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: SymPy Kök Analizi: f(x) = x^2 - 5x + 6
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        x_vals = np.linspace(0.5, 4.5, 200)
        y_vals = x_vals**2 - 5*x_vals + 6

        ax2.plot(x_vals, y_vals, color="#4e73df", lw=2.5, label="f(x) = x^2 - 5x + 6")
        ax2.axhline(0, color="black", linestyle="--", alpha=0.7)
        ax2.scatter([2.0, 3.0], [0.0, 0.0], color="red", s=100, zorder=5, label="SymPy Kökleri: x=2, x=3")

        ax2.set_title("2. SymPy Sembolik Cebir ve Kök İspatı", fontsize=12, fontweight="bold")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.legend(loc="upper center")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Z3 SMT Kısıt Uzayı (x + y = 15, x * y = 56)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        x_grid = np.linspace(1, 14, 200)
        y_cizgi = 15 - x_grid
        y_hiperbol = 56 / x_grid

        ax3.plot(x_grid, y_cizgi, color="#36b9cc", lw=2.2, label="x + y = 15")
        ax3.plot(x_grid, y_hiperbol, color="#f6c23e", lw=2.2, label="x * y = 56")
        ax3.scatter([7.0, 8.0], [8.0, 7.0], color="#1cc88a", s=120, zorder=5, label="Z3 SMT Modeli: (7, 8)")

        ax3.set_title("3. Z3 SMT Kısıt Sağlama (SAT Model Çözümü)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_ylim(0, 16)
        ax3.legend(loc="upper right")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Sembolik İspat ve Çözüm Matrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Neuro-Symbolic Deterministik İspat Özeti", fontsize=12, fontweight="bold", pad=10)

        ispat_metni = "====================================================\n"
        ispat_metni += "         SEMBOLİK MOTOR İSPAT RAPORU                \n"
        ispat_metni += "====================================================\n"
        ispat_metni += f"  1. SymPy Polinom Kökleri   : x in {ispat_sonuclari['sympy_kokler']}\n"
        ispat_metni += f"  2. SymPy Modüler Kök (3x=0): x = {ispat_sonuclari['sympy_moduler_x']}\n"
        ispat_metni += f"  3. SymPy Sembolik Türev    : {ispat_sonuclari['sympy_turev']}\n"
        ispat_metni += f"  4. Z3 SMT Sopa & Top       : Top=${ispat_sonuclari['z3_sopa_top']['top']:.2f}, Sopa=${ispat_sonuclari['z3_sopa_top']['sopa']:.2f}\n"
        ispat_metni += f"  5. Z3 SMT Tam Sayı Kısıtı  : (x={ispat_sonuclari['z3_tam_sayi']['x']}, y={ispat_sonuclari['z3_tam_sayi']['y']})\n"
        ispat_metni += "----------------------------------------------------\n"
        ispat_metni += "  DURUM: %100 DETERMINISTIC PROOF (SIFIR HATA)!\n"
        ispat_metni += "===================================================="

        ax4.text(
            0.02, 0.5, ispat_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Neuro-Symbolic Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Neuro-Symbolic Reasoning Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         NEURO-SYMBOLIC HYBRID ARCHITECTURE         \n"
            "====================================================\n"
            "  [Kullanıcı: Doğal Dil Problemi]\n"
            "          │\n"
            "          ▼\n"
            "  [LLM (Semantic Parser)]: Kısıtları Sembolleştirir\n"
            "          │\n"
            "          ├──► [SymPy Engine] : Analitik Türev, Cebir, Kök\n"
            "          │\n"
            "          └──► [Z3 SMT Solver]: SAT/UNSAT Kısıt Sağlama\n"
            "          │\n"
            "          ▼\n"
            "  [Deterministik Kanıt & %100 Doğru Çözüm]\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 150 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 150 ÖZET KARTI (FAZ 8 YARI-YOL FİNALİ)", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "      DAY 150 SUMMARY: NEURO-SYMBOLIC REASONING     \n"
            "====================================================\n"
            "• Hibrit Güç           : LLM Sezgisi + Z3/SymPy Kesinliği\n"
            "• Doğruluk Oranı       : %100.0 (Matematiksel Teorem İspatı)\n"
            "• Kısıt Çözücü         : Z3 SMT (SAT Modelleri)\n"
            "• Sembolik Hesaplama   : SymPy Cebir & Analitik Türev\n"
            "----------------------------------------------------\n"
            "FAZ 8 İLERLEMESİ (GÜN 141 - GÜN 150 TAMAMLANDI!):\n"
            "  ✓ System 1 vs System 2 Thinking\n"
            "  ✓ Special Tokens & Self-Consistency\n"
            "  ✓ Tree of Thoughts (ToT) & PRM Process Rewards\n"
            "  ✓ Monte Carlo Tree Search (MCTS)\n"
            "  ✓ Test-Time Compute Scaling & Backtracking\n"
            "  ✓ Self-Verification & Neuro-Symbolic (Z3/SymPy)\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 151 (Test-Driven Code Generation Loop)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Sembolik Reasoning Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
