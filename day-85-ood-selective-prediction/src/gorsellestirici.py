"""
OOD Tespiti ve Seçici Tahmin Teşhis ve Görselleştirme Panosu
------------------------------------------------------------
6 panelli yüksek çözünürlüklü Enerji Skoru vs Softmax MSP, AUROC eğrileri,
Kapsam vs Risk (Coverage vs Risk) ve Seçici Tahmin analiz paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class OODGorsellestirici:
    """
    OOD tespit performansını ve seçici tahmin dinamiklerini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_ood_paneli(
        self,
        id_enerji: np.ndarray,
        ood_enerji: np.ndarray,
        metrikler_enerji: Dict[str, Any],
        metrikler_msp: Dict[str, Any],
        kapsam_risk_enerji: Dict[str, np.ndarray],
        esik_degeri: float,
        ham_hata_orani: float,
        filtreli_hata_orani: float,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı OOD ve Seçici Tahmin teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 85: Enerji Tabanlı Dağılım Dışı (OOD) Tespiti ve Seçici Tahmin (Abstention) Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: OOD ve Seçici Tahmin Mimarisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        mimari_metin = (
            "       ENERGY-BASED OOD & SELECTIVE PREDICTION\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. ENERJİ SKORU HESAPLAMA (Liu et al. NeurIPS 2020):\n"
            "     • S_energy(x) = T · log( sum_k exp(z_k / T) )\n"
            "     • Softmax'in normalizasyon baskısını kaldırır.\n"
            "     • Dağılım İçi (ID): Yüksek Enerji Skoru (-E >> 0)\n"
            "     • Dağılım Dışı (OOD): Düşük Enerji Skoru (-E ≈ 0)\n\n"
            "  2. SEÇİCİ TAHMİN / ÇEKİMSERLİK (Abstention):\n"
            "     • Eğer S(x) >= γ  ──> Model Tahmin Eder (OTOMATİK)\n"
            "     • Eğer S(x) <  γ  ──> REDDET & UZMANA DEVRET (GÜVENLİ)\n\n"
            "  3. KAPSAM VS RİSK DENGESİ (Coverage vs Risk):\n"
            "     • Eşik yükseldikçe kabul edilen örneklerde doğruluk %100'e yaklaşır!"
        )
        ax1.text(
            0.5, 0.5, mimari_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Enerji Tabanlı OOD ve Çekimserlik Akışı", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: ID vs OOD Enerji Dağılımı Histogramı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.hist(id_enerji, bins=25, alpha=0.6, color="#38a169", label=f"Dağılım İçi (ID - Ort: {np.mean(id_enerji):.2f})", edgecolor="#22543d")
        ax2.hist(ood_enerji, bins=25, alpha=0.6, color="#e53e3e", label=f"Dağılım Dışı (OOD - Ort: {np.mean(ood_enerji):.2f})", edgecolor="#c53030")
        ax2.axvline(esik_degeri, color="#d69e2e", linestyle="--", linewidth=2.5, label=f"Karar Eşiği γ = {esik_degeri:.2f}")

        ax2.set_title("2. ID vs OOD Enerji Skoru Dağılımı Ayrımı", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.set_xlabel("Enerji Skoru (-E)", fontsize=10)
        ax2.set_ylabel("Örnek Sayısı", fontsize=10)
        ax2.legend(loc="upper left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 3: ROC Eğrisi (Energy vs Softmax MSP AUROC)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(metrikler_enerji["fpr_dizisi"], metrikler_enerji["tpr_dizisi"], "g-", linewidth=2.5, label=f"Enerji Skoru (AUROC: %{metrikler_enerji['auroc']:.2f})")
        ax3.plot(metrikler_msp["fpr_dizisi"], metrikler_msp["tpr_dizisi"], "r--", linewidth=2, label=f"Softmax MSP (AUROC: %{metrikler_msp['auroc']:.2f})")
        ax3.plot([0, 1], [0, 1], "k:", label="Rastgele Tahmin (%50)")

        ax3.set_title("3. OOD Tespit ROC Eğrisi ve AUROC Kıyaslaması", fontsize=12, fontweight="bold", color="#22543d")
        ax3.set_xlabel("False Positive Rate (FPR)", fontsize=10)
        ax3.set_ylabel("True Positive Rate (TPR)", fontsize=10)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1.02)
        ax3.legend(loc="lower right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 4: Kapsam (Coverage) vs Doğruluk / Risk Eğrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        kapsam = kapsam_risk_enerji["kapsam"]
        dogruluk = kapsam_risk_enerji["dogruluk"]
        risk = kapsam_risk_enerji["risk"]

        ax4.plot(kapsam, dogruluk, "b-o", linewidth=2, markersize=4, label="Kabul Edilenlerde Doğruluk (%)")
        ax4.plot(kapsam, risk, "r--s", linewidth=2, markersize=4, label="Kabul Edilenlerde Risk/Hata (%)")

        ax4.set_title("4. Seçici Tahmin: Kapsam (Coverage) vs Doğruluk Dengesi", fontsize=12, fontweight="bold", color="#2c5282")
        ax4.set_xlabel("Kapsam Oranı (% Kabul Edilen Örnekler)", fontsize=10)
        ax4.set_ylabel("Oran (%)", fontsize=10)
        ax4.set_xlim(0, 105)
        ax4.set_ylim(-2, 105)
        ax4.legend(loc="center left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 5: Üretim Güvenlik Eşiğinde Hata Azaltımı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        kategoriler = ["Tüm Örnekler\n(Filtresiz)", "Seçici Tahmin\n(Abstention Aktif)"]
        hata_oranlari = [ham_hata_orani, filtreli_hata_orani]
        bar_colors = ["#e53e3e", "#38a169"]

        bars = ax5.bar(kategoriler, hata_oranlari, color=bar_colors, width=0.45, edgecolor="#2d3748")
        ax5.set_title("5. Seçici Çekimserlik ile Üretim Hata Oranının Düşürülmesi", fontsize=12, fontweight="bold", color="#c53030")
        ax5.set_ylabel("Sınıflandırma Hata Oranı (%)", fontsize=10)
        ax5.set_ylim(0, max(max(hata_oranlari) * 1.35, 10.0))

        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"%{yval:.2f}", ha="center", va="bottom", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       ENERGY OOD & SELECTIVE PREDICTION SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Modeli yeniden eğitmeye gerek yoktur (Pre-trained ile çalışır).\n"
            "  • Softmax MSP'ye göre OOD tespitinde daha yüksek AUROC üretir.\n"
            "  • Canlı ortamda kritik yanlış tahminleri neredeyse sıfıra indirir.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Çekimser kalınan (reddedilen) örnekler insan iş gücü maliyeti yaratır.\n"
            "  • Yakın-OOD (Near-OOD) sınıflarında ayrıştırma zorlaşabilir.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Tıbbi tanı ve otonom araçlarda 'güvenli arıza' (fail-safe) kuralı.\n"
            "  • Bilinmeyen yeni sınıfları otomatik tespit edip etiketleme havuzuna alma.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Eşik çok sıkı seçilirse sistemin otomasyon kapsamı (coverage) çöker."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. OOD & Selective Prediction SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
