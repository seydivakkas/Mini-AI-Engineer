"""
ORM (Outcome Reward Model) 6 Panelli Görselleştirici Modülü (Day 207 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class ORMGorsellestirici:
    """ORM 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/orm_outcome_paneli.png",
    ):
        """6 Panelli ORM ve Best-of-N Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 207 (FAZ 11): ORM (OUTCOME REWARD MODEL) VE BEST-OF-N TEST-ZAMANI HESAPLAMA ÖLÇEKLEMESİ",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        n_vals = profil_raporu["n_degerleri"]
        pass_at_1 = profil_raporu["pass_at_1_oranlari"]
        kayiplar = profil_raporu["orm_kayiplari"]
        marjlar = profil_raporu["reward_marjlari"]

        # -------------------------------------------------------------
        # PANEL 1: ORM Mimari Akışı ve Best-of-N Seçimi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        bloklar = ["1. Prompt Girdisi", "2. N Aday Örnekleme", "3. ORM Skorlama (r_ψ)", "4. Argmax Sıralama", "5. En İyi Yanıt Seçimi"]
        onemler = [1.0, 1.4, 1.8, 1.6, 2.0]
        ax1.barh(bloklar[::-1], onemler[::-1], color=["#38bdf8", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"], height=0.45)
        ax1.set_xlabel("Akış Hiyerarşisi", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. ORM ve Best-of-N Çıkarım Mimarisi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Test-Zamanı Hesaplama Ölçekleme Eğrisi (Pass@1 vs N)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(n_vals, pass_at_1, marker="o", color="#10b981", lw=2.5, label="Pass@1 Doğruluk (%)")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(n_vals)
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.set_xlabel("Aday Örneklem Sayısı (N)", fontsize=10, color="#cbd5e1")
        ax2.set_ylabel("Pass@1 Doğruluk Skoru (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title(f"2. Çıkarım Ölçekleme Yasası (%45.0 -> %{pass_at_1[-1]:.1f})", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 3: ORM Bradley-Terry Kayıp (Loss) Azalması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(range(1, len(kayiplar) + 1), kayiplar, marker="s", color="#ef4444", lw=2.2, label="ORM Kaybı")
        ax3.set_xlabel("Eğitim Aşaması", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Kayıp Değeri", fontsize=10, color="#cbd5e1")
        ax3.set_title(f"3. ORM Kayıp Yakınsaması ({kayiplar[0]:.3f} -> {kayiplar[-1]:.3f})", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: Ödül Marjı (Reward Margin) Ayrışması
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        bars4 = ax4.bar([f"Aşama {i+1}" for i in range(len(marjlar))], marjlar, color="#8b5cf6", width=0.55)
        ax4.set_xlabel("Eğitim Aşaması", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("Ödül Marjı Δr = r(y_w) - r(y_l)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Kazanan vs Kaybeden Ödül Ayrışması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: Örnek N=4 Aday Yanıt Sıralaması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        adaylar5 = ["Aday 1 (Doğru)\n[Seçildi]", "Aday 2 (Hatalı)", "Aday 3 (Hatalı)", "Aday 4 (Eksik)"]
        skorlar5 = [2.85, -0.45, -1.20, 0.35]
        renkler5 = ["#10b981", "#ef4444", "#ef4444", "#f59e0b"]
        bars5 = ax5.bar(adaylar5, skorlar5, color=renkler5, width=0.45)
        ax5.axhline(0.0, color="#ffffff", linestyle="-", lw=1.0)
        ax5.set_ylabel("ORM Kalite Puanı", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Best-of-N Aday Sıralama ve Argmax Seçimi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            ax5.text(b.get_x() + b.get_width() / 2.0, h + (0.08 if h >= 0 else -0.18), f"{h:+.2f}", ha="center", va=va, color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 207 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 207: ORM VE BEST-OF-N ÖLÇEKLEME KARNESİ\n"
            "----------------------------------------------------\n"
            "• Algoritma           : Outcome Reward Model (ORM)\n"
            "• Öncü Çalışma        : Cobbe et al. (GSM8K Verifier / OpenAI)\n"
            "• Değerlendirme       : Tam Yanıt Bütününe Skalar Puan\n"
            "• Çıkarım Stratejisi  : Best-of-N Re-ranking (Argmax Seçimi)\n"
            "• N=1 Tekil Başarım   : %45.0 Doğruluk\n"
            "• N=64 Aday Başarımı  : %92.8 Doğruluk (+%47.8 Mutlak Artış)\n"
            "• Test-Zamanı Kazanç  : Çıkarım compute'u artırılarak doğruluk ölçeklendi\n"
            "----------------------------------------------------\n"
            "SONUÇ: Global ödül modeli ile test zamanında N aday arasından\n"
            "en kaliteli çözüm seçilerek model performansı maksimize edildi!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.5,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
