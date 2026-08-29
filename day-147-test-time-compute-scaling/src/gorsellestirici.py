"""
Test-Time Compute Scaling Teşhis Panosu Görselleştirici Modülü (Day 147 - Faz 8).
6 panelli Power-Law Scaling Eğrisi, Pareto Sınırı (8B vs 70B), Derinlik vs Genişlik, Maliyet Tablosu, Akış Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class TestTimeScalingGorsellestirici:
    """Test-time compute scaling teşhis panosu üretir."""

    __test__ = False  # PyTest'in test sınıfı olarak toplamasını engeller

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        scaling_verileri: List[Dict[str, Any]],
        pareto_verileri: List[Dict[str, Any]],
        butce_analizi: Dict[str, Any],
        kayit_yolu: str = "ciktilar/test_time_compute_scaling_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 147: Test-Time Compute Scaling Yasaları: Çıkarım Zamanı Hesaplama Bütçesi & Pareto Sınırları",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Power-Law Scaling Doğruluk Eğrisi (1x - 256x)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        butceler = [s["butce_n"] for s in scaling_verileri]
        dogruluklar = [s["dogruluk_orani"] * 100 for s in scaling_verileri]

        ax1.plot(butceler, dogruluklar, marker="o", color="#4e73df", lw=2.5, label="Test-Time Scaling Eğrisi")
        for b, acc in zip(butceler, dogruluklar):
            ax1.text(b, acc + 2.0, f"%{acc:.1f}", ha="center", fontsize=9.5, fontweight="bold")

        ax1.set_xscale("log", base=2)
        ax1.set_title("1. Test-Time Compute Scaling Yasası (Power-Law)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Çıkarım Hesaplama Bütçesi (N Katsayısı - Log2)")
        ax1.set_ylabel("Akıl Yürütme Doğruluğu (%)")
        ax1.set_ylim(30, 105)
        ax1.legend(loc="lower right")
        ax1.grid(True, which="both", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Pareto Sınırı (8B + Test Compute vs 70B)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        maliyet_8b = [p["maliyet_birimi"] for p in pareto_verileri if p["model"] == "8B"]
        acc_8b = [p["dogruluk"] * 100 for p in pareto_verileri if p["model"] == "8B"]

        maliyet_70b = [p["maliyet_birimi"] for p in pareto_verileri if p["model"] == "70B"]
        acc_70b = [p["dogruluk"] * 100 for p in pareto_verileri if p["model"] == "70B"]

        ax2.plot(maliyet_8b, acc_8b, marker="s", color="#1cc88a", lw=2.2, label="8B + Test-Time Compute")
        ax2.plot(maliyet_70b, acc_70b, marker="^", color="#e74a3b", lw=2.2, label="70B + Test-Time Compute")

        ax2.set_xscale("log")
        ax2.set_title("2. Pareto Verimlilik Sınırı: 8B vs 70B", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Göreli Çıkarım Maliyeti (Log Ölçek)")
        ax2.set_ylabel("Doğruluk Oranı (%)")
        ax2.set_ylim(45, 102)
        ax2.legend(loc="lower right")
        ax2.grid(True, which="both", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Arama Stratejileri Doğruluk Kıyası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        strat_isimleri = ["Paralel Örnekleme\n(Self-Consistency)", "Derin Sıralı Arama\n(Tekil Uzun Zincir)", "Dengeli MCTS\n(Ağaç Araması)"]
        strat_acc = [
            butce_analizi["paralel_ornekleme"]["tahmini_dogruluk"] * 100,
            butce_analizi["derin_sirali_arama"]["tahmini_dogruluk"] * 100,
            butce_analizi["dengeli_agac_aramasi"]["tahmini_dogruluk"] * 100,
        ]
        renkler3 = ["#36b9cc", "#f6c23e", "#1cc88a"]

        barlar3 = ax3.bar(strat_isimleri, strat_acc, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

        ax3.set_title("3. Sabit Bütçede (4096 Token) Arama Stratejileri", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Tahmini Doğruluk (%)")
        ax3.set_ylim(0, 105)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Token Bütçesi ve Çıkarım Maliyet Tablosu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Test-Time Compute Bütçe ve Doğruluk Matrisi", fontsize=12, fontweight="bold", pad=10)

        tablo_metni = "====================================================\n"
        tablo_metni += "BÜTÇE (N) | HATA ORANI | DOĞRULUK | KAZANIM KATSAYISI \n"
        tablo_metni += "====================================================\n"
        for row in scaling_verileri:
            tablo_metni += f"{row['butce_n']:<9} | %{row['hata_orani']*100:<10.1f} | %{row['dogruluk_orani']*100:<8.1f} | {row['hesaplama_kat_artisi']:<17}\n"
        tablo_metni += "====================================================\n"
        tablo_metni += "Kritik Çıkarım: N=1 -> N=64 geçişinde doğruluk\n"
        tablo_metni += "%30.0'dan %88.0+'a sıçrayarak log-lineer scaling gösterir!"

        ax4.text(
            0.02, 0.5, tablo_metni,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Test-Time Compute Scaling Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Test-Time Compute Scaling Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "       TEST-TIME COMPUTE PARADIGM SHIFT             \n"
            "====================================================\n"
            "  [Eski Paradigma: Pre-training Scaling]\n"
            "  • Daha akıllı model = Daha çok parametre (70B, 405B)\n"
            "  • Devasa eğitim maliyeti, statik $O(1)$ çıkarım.\n\n"
            "  [Yeni Paradigma: Test-Time Compute Scaling]\n"
            "  • Sabit parametreli model (örn: 8B veya 32B)\n"
            "  • Zor sorularda daha fazla düşünme bütçesi (N_tokens)\n"
            "  • Arama ağaçları (MCTS/ToT) + PRM doğrulayıcılar\n"
            "  • 8B + 16x Compute == 70B Standart Model!\n"
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
        # PANEL 6: GÜN 147 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 147 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "      DAY 147 SUMMARY: TEST-TIME COMPUTE SCALING    \n"
            "====================================================\n"
            "• Power-Law Scaling    : Hata(N) = alpha * N^(-beta) + gamma\n"
            "• Pareto Verimliliği   : 8B + 16x Compute >= 70B Base\n"
            "• Optimal Strateji     : Dengeli MCTS (%96 doğruluk)\n"
            "• Yeni AI Çağı         : Eğitimden Çıkarım Zamanı Zekasına\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. o1 ve DeepSeek-R1 test-time compute teorik temeli\n"
            "  2. Derinlik vs Genişlik bütçe optimizasyonu\n"
            "  3. Pareto sınırında küçük modelleri süper-modelleştirme\n"
            "  4. Çıkarım gecikmesi ve maliyet optimizasyon formülü\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 148 (Geri İzleme & Hata İyileştirme)\n"
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
        print(f"  ✓ Test-Time Compute Scaling Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
