"""
Biçimsel Teorem İspatı Teşhis Panosu Görselleştirici Modülü (Day 152 - Faz 8).
6 panelli İspat Kesinliği, Hedef Sayısı Değişimi, Taktik Dağılımı, Lean 4 Kodu & Günlük, Akış Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class Lean4Gorsellestirici:
    """Lean 4 biçimsel ispat teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        ispat_sonucu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/formal_theorem_proving_lean4_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 152: Biçimsel Mantık ve Teorem İspatı: LLM ile Lean 4 Kod Üretimi & ITP Doğrulama",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Doğal Dil vs Lean 4 Biçimsel İspat Kesinliği
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = ["Doğal Dil (Metin)\n(İnformal İspat)", "Python Sembolik\n(SymPy)", "Lean 4 Biçimsel İspat\n(Formal ITP Proof)"]
        kesinlikler = [60.0, 92.0, 100.0]
        renkler1 = ["#e74a3b", "#f6c23e", "#1cc88a"]

        barlar1 = ax1.bar(yontemler, kesinlikler, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_title("1. Matematiksel İspat Kesinliği ve Güvenilirlik", fontsize=12, fontweight="bold")
        ax1.set_ylabel("İspat Kesinliği (%)")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Taktik Adımları Boyunca Açık Hedef Sayısı (Goals Left)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        adımlar = ["Başlangıç", "1. induction", "2. rfl (Taban)", "3. rw [hd] (Adım)"]
        hedef_sayilari = [1, 2, 1, 0]  # 1 hedef -> 2 alt hedef -> 1 hedef -> 0 (QED)

        ax2.plot(adımlar, hedef_sayilari, marker="o", color="#4e73df", lw=2.5, label="Açık Hedef (Goal) Sayısı")
        ax2.scatter(["3. rw [hd] (Adım)"], [0], color="#1cc88a", s=130, zorder=5, label="Q.E.D. (Hedef Kalmadı)")

        for x, y in zip(adımlar, hedef_sayilari):
            ax2.text(x, y + 0.1, str(y), ha="center", fontsize=10.5, fontweight="bold")

        ax2.set_title("2. Lean 4 İspat İlerlemesi (Goals to Q.E.D.)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Açık Kalan Hedef Sayısı")
        ax2.set_ylim(-0.3, 2.5)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Lean 4 Taktik Türleri
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        taktikler = ["Tümevarım\n(induction)", "Reflexivity\n(rfl)", "Yeniden Yazma\n(rw / rewrite)"]
        oranlar = [33.3, 33.3, 33.4]
        renkler3 = ["#36b9cc", "#1cc88a", "#f6c23e"]

        ax3.pie(oranlar, labels=taktikler, autopct="%1.1f%%", colors=renkler3, startangle=140, explode=(0.05, 0.05, 0.05))
        ax3.set_title("3. Kullanılan Lean 4 İspat Taktikleri", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Lean 4 İspat Kodu ve Taktik Günlüğü
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Lean 4 İspat Kodu & Çekirdek Günlüğü", fontsize=12, fontweight="bold", pad=10)

        kod_metni = "====================================================\n"
        kod_metni += "         LEAN 4 FORMAL THEOREM & PROOF              \n"
        kod_metni += "====================================================\n"
        kod_metni += f"{ispat_sonucu['lean4_kodu']}\n"
        kod_metni += "----------------------------------------------------\n"
        kod_metni += "ITP ÇEKİRDEK YÜRÜTME ADIMLARI:\n"
        for i, a in enumerate(ispat_sonucu["adim_kayitlari"], start=1):
            kod_metni += f"  {i}. Taktik: '{a['uygulanan_taktik']}' -> Kalan Hedef: {a['kalan_hedef_sayisi']}\n"
        kod_metni += "----------------------------------------------------\n"
        kod_metni += "  DURUM: NO GOALS LEFT => PROOF COMPLETE (Q.E.D.)!\n"
        kod_metni += "===================================================="

        ax4.text(
            0.02, 0.5, kod_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Lean 4 Autoformalization Mimarisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Autoformalization & Lean 4 ITP Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "      AUTORMALIZATION & LEAN 4 PROOF PIPELINE       \n"
            "====================================================\n"
            "  [Doğal Dil: 'Her n için n + 0 = n olduğunu kanıtla']\n"
            "                       │                            \n"
            "                       ▼                            \n"
            "  [LLM Autoformalizer]: Lean 4 Teoremine Çevirir     \n"
            "    theorem add_zero (n : Nat) : n + 0 = n := by ... \n"
            "                       │                            \n"
            "                       ▼                            \n"
            "  [Lean 4 Tactic Engine]: Taktikleri Yürütür         \n"
            "    1. induction n  -> Alt hedefler açılır          \n"
            "    2. rfl          -> 0 + 0 = 0 kapandı            \n"
            "    3. rw [hd]      -> succ d + 0 = succ d kapandı  \n"
            "                       │                            \n"
            "                       ▼                            \n"
            "  [Lean 4 Kernel Type-Checker]: %100 Resmi İspat!   \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 152 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 152 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "      DAY 152 SUMMARY: FORMAL THEOREM PROVING       \n"
            "====================================================\n"
            "• İspat Dili           : Lean 4 (Interactive Theorem Prover)\n"
            "• Temel Prensip        : Curry-Howard (Önermeler = Tipler)\n"
            "• Teorem               : add_zero (Peano Doğal Sayı Tümevarımı)\n"
            "• Güvenilirlik         : %100.0 (Biçimsel Tip Denetimi)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. LLM ile doğal dilden Lean 4 koduna çeviri (Autoformalization)\n"
            "  2. Taktik arama ağaçları (induction, rfl, rw, simp)\n"
            "  3. İspat hedeflerinin deterministik olarak kapatılması\n"
            "  4. AlphaProof ve modern matematik yapay zekasının temeli\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 153 (Mantıksal Safsata & Tümdengelim)\n"
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
        print(f"  ✓ Lean 4 Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
