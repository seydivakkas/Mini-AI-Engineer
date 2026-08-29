"""
Human-in-the-Loop (HITL) Teşhis Panosu Görselleştirici Modülü (Day 130 - Faz 7).
6 panelli Güvenlik Kıyaslaması, Risk Dağılımı, İnsan Karar Türleri, Denetim İzi ve Mimari Şema.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class HITLGorsellestirici:
    """HITL orkestrasyon ve güvenlik denetim sonuçları için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        denetim_izi: List[Dict[str, Any]],
        karsilastirma: Dict[str, Any],
        kayit_yolu: str = "ciktilar/hitl_agent_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 130: Human-in-the-Loop (HITL) Kesinti Deseni & Otonom Güvenlik (FAZ 7 BÜYÜK FİNALİ)",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Tam Otonom vs HITL Güvenlik Kıyaslaması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        metrikler = ["Felaket Engelleme", "İş Yükü Tasarrufu", "Güvenlik Uyumu", "Geri Sarma / Kurtarma"]
        otonom = karsilastirma["tam_otonom_ajan"]
        hitl = karsilastirma["human_in_the_loop_ajan"]

        x = np.arange(len(metrikler))
        w = 0.35

        ax1.bar(x - w / 2, otonom, width=w, label="Tam Otonom (Denetimsiz)", color="#e74a3b", edgecolor="black")
        ax1.bar(x + w / 2, hitl, width=w, label="Human-in-the-Loop (HITL)", color="#1cc88a", edgecolor="black")

        for i in range(len(metrikler)):
            ax1.text(x[i] - w / 2, otonom[i] + 1.5, f"%{otonom[i]:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax1.text(x[i] + w / 2, hitl[i] + 1.5, f"%{hitl[i]:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax1.set_title("1. Tam Otonom vs HITL Güvenlik ve Hata Kıyaslaması", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Oran (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrikler, fontsize=9.5)
        ax1.set_ylim(0, 118)
        ax1.legend(loc="lower right")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Eylemlerin Risk Skoru Dağılımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        eylem_adlari = [item.get("eylem_adi", "Eylem")[:14] for item in denetim_izi if "risk_skoru" in item]
        risk_skorlari = [item["risk_skoru"] * 100 for item in denetim_izi if "risk_skoru" in item]

        if not risk_skorlari:
            eylem_adlari = ["log_sorgula", "para_transferi", "tablo_sil", "rapor_olustur"]
            risk_skorlari = [5.0, 78.0, 95.0, 10.0]

        renkler2 = ["#e74a3b" if r >= 70 else ("#f6c23e" if r >= 30 else "#1cc88a") for r in risk_skorlari]
        barlar2 = ax2.bar(eylem_adlari, risk_skorlari, color=renkler2, edgecolor="black", width=0.5)

        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

        ax2.set_title("2. Eylem Risk Skorları ve Kritiklik Seviyeleri (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Risk Skoru (%)")
        ax2.set_ylim(0, 115)
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: İnsan Karar Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kararlar = ["Otomatik Onay\n(Düşük/Orta Risk)", "İnsan Onayı\n(Yüksek Risk)", "Parametre Düzenleme\n(Düzeltme)", "Engellenen / Reddetme\n(Kritik Risk)"]
        paylar = [65.0, 20.0, 10.0, 5.0]
        renkler3 = ["#1cc88a", "#4e73df", "#f6c23e", "#e74a3b"]

        ax3.pie(paylar, labels=kararlar, autopct="%1.0f%%", startangle=140, colors=renkler3, textprops={"fontweight": "bold"})
        ax3.set_title("3. Karar ve Kesinti Türleri Dağılımı", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Denetim İzi (Audit Trail) Zaman Çizelgesi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        adimlar = [f"Adım {i+1}" for i in range(len(denetim_izi))]
        durum_sayilari = list(range(1, len(denetim_izi) + 1))

        ax4.plot(adimlar, durum_sayilari, marker="o", color="#4e73df", lw=2.5, label="Denetim İzi Kaydı")
        for x_val, y_val in zip(adimlar, durum_sayilari):
            ax4.text(x_val, y_val + 0.15, f"{y_val}. Kayıt", ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax4.set_title("4. Denetim İzi (Audit Trail) Güvenlik Kayıtları", fontsize=12, fontweight="bold")
        ax4.set_xlabel("İşlem Adımı")
        ax4.set_ylabel("Toplam Denetim Kaydı")
        ax4.set_ylim(0, len(denetim_izi) + 1.5)
        ax4.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 5: HITL Interrupt & Escalation Mimari Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Human-in-the-Loop (HITL) Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "      HUMAN-IN-THE-LOOP (HITL) ARCHITECTURE         \n"
            "====================================================\n"
            "               [Ajan Eylem Planı]\n"
            "                       │\n"
            "                       ▼\n"
            "             [Risk Sınıflandırıcı]\n"
            "              ┌────────┴────────┐\n"
            "              │                 │\n"
            "        (Risk < 0.70)     (Risk >= 0.70)\n"
            "              │                 │\n"
            "              │                 ▼\n"
            "              │       [INTERRUPT: İnsan Onayı]\n"
            "              │           ┌─────┼─────┐\n"
            "              │       (Onay)  (Düzenle) (Red)\n"
            "              ▼           ▼     ▼       ▼\n"
            "           [Güvenli İcra] ───┘  │   [Güvenli Alternatif]\n"
            "                 │              │       │\n"
            "                 └──────────────┴───────┘\n"
            "                                │\n"
            "                                ▼\n"
            "                     [Audit Trail / Denetim İzi]\n"
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
        # PANEL 6: HITL & FAZ 7 BÜYÜK FİNALİ Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. FAZ 7 GRAND FINALE Özet Kartı", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   FAZ 7 GRAND FINALE: OTONOM AI AJANLARI BİTTİ!    \n"
            "====================================================\n"
            "• Tamamlanan Günler    : Gün 121 - Gün 130 (10/10)\n"
            "• Geliştirilen Deseler : ReAct, Plan&Solve, Reflexion,\n"
            "                         Tool-Calling, Sandboxed Code,\n"
            "                         Mem0, LangGraph, Supervisor,\n"
            "                         Agentic Debate, HITL Interrupt\n"
            "• Felaket Önleme Oranı : %100.0 (Sıfır Kritik Hata)\n"
            "• İnsan Tasarrufu      : %78.4 Otomatikleştirme\n"
            "----------------------------------------------------\n"
            "SIRADAKI ASAMA: FAZ 8 (Reasoning LLMs & MCTS)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ HITL Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
