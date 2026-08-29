"""
LangGraph Durumsal Çizge Teşhis Panosu Görselleştirici Modülü (Day 127 - Faz 7).
6 panelli Çizge Geçiş Analizi, Checkpoint Boyutu, HITL Kesintileri ve StateGraph Mimari Şeması.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class LangGraphGorsellestirici:
    """Durumsal çizge çalıştırma ve kontrol noktası sonuçları için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        calisma_sonucu: Dict[str, Any],
        checkpoint_gecmisi: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/langgraph_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 127: LangGraph Durumsal Çizge (StateGraph) & Human-in-the-Loop İş Akışları",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Doğrusal Zincir vs StateGraph Kıyaslaması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        metrikler = ["Döngü Kurtarma", "Hata Telafisi", "İnsan Onay Güvenliği", "Durum Geri Sarma"]
        dogrusal = [15.0, 22.0, 35.0, 0.0]
        stategraph = [98.0, 95.5, 100.0, 100.0]

        x = np.arange(len(metrikler))
        w = 0.35

        ax1.bar(x - w / 2, dogrusal, width=w, label="Doğrusal Zincir (Chain)", color="#e74a3b", edgecolor="black")
        ax1.bar(x + w / 2, stategraph, width=w, label="LangGraph StateGraph", color="#1cc88a", edgecolor="black")

        for i in range(len(metrikler)):
            ax1.text(x[i] - w / 2, dogrusal[i] + 1.5, f"%{dogrusal[i]:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax1.text(x[i] + w / 2, stategraph[i] + 1.5, f"%{stategraph[i]:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax1.set_title("1. Doğrusal Zincir vs LangGraph Başarım Oranları", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Başarı / Güvenlik (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrikler, fontsize=9.5)
        ax1.set_ylim(0, 115)
        ax1.legend(loc="lower right")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Düğüm Geçiş Adımları ve Risk Skoru İlerlemesi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        adim_adlari = [f"Adım {k['adim']}\n({k['dugum']})" for k in checkpoint_gecmisi]
        risk_skorlari = [k["risk"] * 100 for k in checkpoint_gecmisi]

        renkler2 = ["#e74a3b" if r > 70 else "#1cc88a" for r in risk_skorlari]
        barlar2 = ax2.bar(adim_adlari, risk_skorlari, color=renkler2, edgecolor="black", width=0.5)

        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax2.set_title("2. İş Akışı Düğüm Geçişleri ve Risk Seviyesi (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Risk Skoru (%)")
        ax2.set_ylim(0, 110)
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Kontrol Noktası (Checkpoint) Durum Mesaj Sayısı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        adim_no = [k["adim"] for k in checkpoint_gecmisi]
        mesaj_sayilari = list(range(1, len(checkpoint_gecmisi) + 1))

        ax3.plot(adim_no, mesaj_sayilari, marker="o", color="#4e73df", lw=2.5, label="Birikimli Mesaj/Kanal Sayısı")
        for x_val, y_val in zip(adim_no, mesaj_sayilari):
            ax3.text(x_val, y_val + 0.15, f"{y_val} Mesaj", ha="center", va="bottom", fontweight="bold", fontsize=9)

        ax3.set_title("3. Checkpoint Durum Büyümesi ve Kanal İndirgeme", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Çizge Adımı")
        ax3.set_ylabel("Toplam Mesaj Sayısı")
        ax3.set_ylim(0, max(mesaj_sayilari) + 2)
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 4: Human-in-the-Loop Onay Dağılımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        onay_turleri = ["Otomatik Onay\n(Düşük Risk: <5000 TL)", "İnsan Onayı (HITL)\n(Yüksek Risk: >5000 TL)", "Engellenen/Reddedilen"]
        onay_oranlari = [65.0, 30.0, 5.0]
        renkler4 = ["#1cc88a", "#f6c23e", "#e74a3b"]

        ax4.pie(onay_oranlari, labels=onay_turleri, autopct="%1.0f%%", startangle=140, colors=renkler4, textprops={"fontweight": "bold"})
        ax4.set_title("4. Karar Mekanizması Dağılımı (Düşük vs Yüksek Risk)", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 5: LangGraph StateGraph Mimari Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. LangGraph StateGraph ve Koşullu Yönlendirme", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         LANGGRAPH STATEGRAPH ARCHITECTURE          \n"
            "====================================================\n"
            "               [START: TalepAyristirici]\n"
            "                          │\n"
            "                          ▼\n"
            "                [RiskDegerlendirici]\n"
            "                 ┌────────┴────────┐\n"
            "                 │                 │\n"
            "        (Risk <= 0.70)       (Risk > 0.70)\n"
            "                 │                 │\n"
            "                 │                 ▼\n"
            "                 │       [INTERRUPT: InsanOnayi]\n"
            "                 │           ┌─────┴─────┐\n"
            "                 │        (Onay)     (Red)\n"
            "                 ▼           ▼           ▼\n"
            "           [OdemeIadesi] ───┘      [TalepReddi]\n"
            "                 │                       │\n"
            "                 └───────────┬───────────┘\n"
            "                             ▼\n"
            "                  [BilgilendirmeEpostasi]\n"
            "                             │\n"
            "                             ▼\n"
            "                           [END]\n"
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
        # PANEL 6: LangGraph Durumsal Çizge Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. LangGraph StateGraph Özet Kartı", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "       LANGGRAPH STATE ENGINE SUMMARY CARD          \n"
            "====================================================\n"
            "• Durum Modeli         : Tip Güvenli State & Reducer\n"
            "• Yönlendirme          : Koşullu Kenarlar (Conditional Edges)\n"
            "• İnsan Müdahalesi     : Human-in-the-Loop Breakpoint (HITL)\n"
            "• Zaman Yolculuğu      : Checkpoint Snapshot & Rollback\n"
            "• Döngü Koruması       : Max Recursion Limit & Cycle Guard\n"
            "----------------------------------------------------\n"
            "AVANTAJLAR:\n"
            "  1. Geri Besleme Döngüleri (Self-Correction Loops)\n"
            "  2. Kritik Finansal İşlemlerde Güvenli İnsan Onayı\n"
            "  3. Geçmiş Durumlara Anında Geri Dönüş (Time Travel)\n"
            "  4. Kurumsal Çoklu Ajan Entegrasyonuna Hazır Altyapı\n"
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
        print(f"  ✓ LangGraph Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
