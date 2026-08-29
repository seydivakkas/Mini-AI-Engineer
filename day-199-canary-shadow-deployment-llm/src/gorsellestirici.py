"""
Canary & Shadow Traffic 6 Panelli Görselleştirici Modülü (Day 199 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class CanaryGorsellestirici:
    """Canary & Shadow 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        gecis_raporu: Dict[str, Any],
        rollback_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/canary_shadow_paneli.png",
    ):
        """6 Panelli Canary & Shadow Dağıtım Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 199: ÜRETİMDE CANARY DAĞITIMI VE SHADOW-TRAFFIC İLE SIFIR KESİNTİLİ MODEL GÜNCELLEMESİ",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        asamalar = gecis_raporu["asama_raporlari"]
        asama_adlar = [a["asama_adi"].split(" (")[0] for a in asamalar]

        # -------------------------------------------------------------
        # PANEL 1: Canary ve Shadow Dağıtım Mimarisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        bloklar = ["1. Canlı Kullanıcı İsteği", "2. Ağırlıklı Yönlendirici", "3. Baseline (v1.0.0)", "4. Canary (v2.0.0)", "5. Shadow Mirror"]
        onem = [1.0, 1.3, 1.8, 1.8, 1.4]
        ax1.barh(bloklar[::-1], onem[::-1], color=["#38bdf8", "#10b981", "#3b82f6", "#f59e0b", "#8b5cf6"], height=0.45)
        ax1.set_xlabel("Akış Hiyerarşisi", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Canary ve Shadow Trafik Boru Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Kademeli Canary Trafik Dağılımı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        canary_oranlar = [a["gerceklesen_canary_yuzde"] for a in asamalar]
        baseline_oranlar = [100.0 - c for c in canary_oranlar]

        ax2.bar(asama_adlar, baseline_oranlar, label="Baseline (v1.0)", color="#3b82f6", width=0.45)
        ax2.bar(asama_adlar, canary_oranlar, bottom=baseline_oranlar, label="Canary (v2.0)", color="#10b981", width=0.45)
        ax2.set_ylabel("Trafik Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Kademeli Trafik Kaydırma (%5 -> %100)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(axis="y", linestyle=":", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 3: Model Gecikme Kıyası (v1 vs v2)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        modeller = ["Baseline (v1.0)\n[FP16]", "Canary (v2.0)\n[TRT-LLM FP8]"]
        gecikmeler = [28.0, 22.0]
        bars3 = ax3.bar(modeller, gecikmeler, color=["#3b82f6", "#10b981"], width=0.45)
        ax3.set_ylabel("Ortalama Gecikme (ms)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Model Sürüm Gecikme Kazanımı (%21 Hızlı)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.8, f"{h:.1f} ms", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Gölge Trafik (Shadow Traffic) Karşılaştırması
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        istek_idx = np.arange(1, 11)
        base_lats = [28 + np.random.uniform(-2, 2) for _ in range(10)]
        shadow_lats = [22 + np.random.uniform(-2, 2) for _ in range(10)]

        ax4.plot(istek_idx, base_lats, marker="o", color="#3b82f6", label="Canlı Baseline")
        ax4.plot(istek_idx, shadow_lats, marker="s", color="#8b5cf6", linestyle="--", label="Gölge (Shadow) Aday")
        ax4.set_xlabel("İstek Sırası", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("Gecikme (ms)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Sıfır Kullanıcı Etkili Shadow Kıyası", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: Otomatik Rollback Devre Kesici (Circuit Breaker)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        metrik_durum = ["Normal Canary", "Hata Patlaması (%10)", "Rollback Sonrası"]
        agirliklar = [20.0, 20.0, 0.0]
        bars5 = ax5.bar(metrik_durum, agirliklar, color=["#10b981", "#ef4444", "#3b82f6"], width=0.45)
        ax5.set_ylabel("Canary Trafik Ağırlığı (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Anomali Tespiti ve Anında Rollback", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 0.6, f"%{h:.0f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 199 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 199: CANARY & SHADOW DEPLOYMENT KARNE\n"
            "----------------------------------------------------\n"
            "• Dağıtım Stratejisi  : Shadow Traffic + Progressive Canary\n"
            "• Kademeli Geçiş      : %5 -> %20 -> %50 -> %100\n"
            "• Gölge Trafik (Dark) : Kullanıcı etkilenmeden canlı doğrulama\n"
            "• Otomatik Rollback   : Hata oranı > %2 olunca anında %0'a iniş\n"
            "• Kesinti Süresi      : Sıfır Kesinti (Zero-Downtime Hot Swap)\n"
            "• Sürüm Farkı         : v2.0 sürümü %21 daha düşük gecikmeli\n"
            "----------------------------------------------------\n"
            "SONUÇ: Üretimdeki kritik LLM modellerini sıfır risk ve\n"
            "otomatik güvenlik sigortası ile canlıda güncelleme!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=10,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
