"""
Olasılık Kalibrasyonu Teşhis ve Görselleştirme Panosu
----------------------------------------------------
6 panelli yüksek çözünürlüklü güvenilirlik diyagramları (Reliability Diagrams),
ECE/MCE metrikleri, güven histogramları ve Sıcaklık Ölçekleme (Temperature Scaling) analiz paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


class KalibrasyonGorsellestirici:
    """
    Model kalibrasyonunu ve güvenilirlik diyagramlarını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_kalibrasyon_paneli(
        self,
        onceki_metrikler: Dict[str, Any],
        sonraki_metrikler: Dict[str, Any],
        optimal_sicaklik: float,
        sicaklik_tarama: Dict[str, np.ndarray],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Kalibrasyon ve ECE teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 84: Olasılık Kalibrasyonu, Expected Calibration Error (ECE) & Temperature Scaling Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Kalibrasyon Kavramı ve Aşırı Güven (Overconfidence)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        kavram_metin = (
            "       OLASILIK KALİBRASYONU (PROBABILITY CALIBRATION)\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. AŞIRI GÜVEN PROBLEMİ (Overconfidence Crisis):\n"
            "     • Modern derin ağlar (ResNet, ViT) %99 güvenle tahmin\n"
            "       yaptığı durumlarda gerçekte sadece %70 doğru çıkabilir!\n"
            "     • Otonom sürüş ve tıp için bu durum kabul edilemez.\n\n"
            "  2. İDEAL KALİBRASYON (Perfect Calibration):\n"
            "     • P( Doğru Tahmin | Güven = p ) = p\n"
            "     • Model '%80 eminim' diyorsa 100 örnekten 80'i doğru olmalıdır.\n\n"
            "  3. POST-HOC TEMPERATURE SCALING (Guo et al. 2017):\n"
            "     • q_i = Softmax(z_i / T*)\n"
            "     • Model doğruluğu ve tahmin sırası %100 AYNI KALIR!\n"
            "     • Yalnızca güven skorları kalibre edilir (ECE düşürülür)."
        )
        ax1.text(
            0.5, 0.5, kavram_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Model Kalibrasyonu ve Güvenilirlik İlkesi", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Kalibrasyon Öncesi Güvenilirlik Diyagramı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        bins = onceki_metrikler["bin_guvenleri"]
        accs_once = onceki_metrikler["bin_dogruluklari"]
        gaps_once = onceki_metrikler["bin_farklari"]

        ax2.plot([0, 1], [0, 1], "k--", label="İdeal Kalibrasyon (y = x)")
        ax2.bar(bins, accs_once, width=1.0/len(bins), alpha=0.7, color="#3182ce", edgecolor="#2b6cb0", label="Gözlenen Doğruluk")
        ax2.bar(bins, gaps_once, bottom=accs_once, width=1.0/len(bins), alpha=0.4, color="#e53e3e", edgecolor="#c53030", hatch="//", label="Kalibrasyon Açığı (Gap)")

        ax2.set_title(f"2. Kalibrasyon Öncesi (ECE: %{onceki_metrikler['ece']:.2f} | NLL: {onceki_metrikler['nll']:.3f})", fontsize=12, fontweight="bold", color="#c53030")
        ax2.set_xlabel("Tahmin Güveni (Confidence)", fontsize=10)
        ax2.set_ylabel("Doğruluk (Accuracy)", fontsize=10)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 3: Kalibrasyon Sonrası Güvenilirlik Diyagramı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        bins_sonra = sonraki_metrikler["bin_guvenleri"]
        accs_sonra = sonraki_metrikler["bin_dogruluklari"]
        gaps_sonra = sonraki_metrikler["bin_farklari"]

        ax3.plot([0, 1], [0, 1], "k--", label="İdeal Kalibrasyon (y = x)")
        ax3.bar(bins_sonra, accs_sonra, width=1.0/len(bins_sonra), alpha=0.7, color="#38a169", edgecolor="#22543d", label="Kalibre Doğruluk")
        ax3.bar(bins_sonra, gaps_sonra, bottom=accs_sonra, width=1.0/len(bins_sonra), alpha=0.4, color="#e53e3e", edgecolor="#c53030", hatch="//", label="Kalan Açık (Gap)")

        ax3.set_title(f"3. Kalibrasyon Sonrası (T* = {optimal_sicaklik:.2f} | ECE: %{sonraki_metrikler['ece']:.2f} | NLL: {sonraki_metrikler['nll']:.3f})", fontsize=12, fontweight="bold", color="#22543d")
        ax3.set_xlabel("Kalibre Tahmin Güveni", fontsize=10)
        ax3.set_ylabel("Doğruluk (Accuracy)", fontsize=10)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.legend(loc="upper left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 4: Güven Dağılımı Histogramı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.hist(onceki_metrikler["tum_guvenler"], bins=20, alpha=0.5, color="#e53e3e", label=f"Ham Model (Ort. Güven: {np.mean(onceki_metrikler['tum_guvenler']):.2f})", edgecolor="#c53030")
        ax4.hist(sonraki_metrikler["tum_guvenler"], bins=20, alpha=0.6, color="#38a169", label=f"Kalibre Model (Ort. Güven: {np.mean(sonraki_metrikler['tum_guvenler']):.2f})", edgecolor="#22543d")

        ax4.set_title("4. Tahmin Güveni Dağılımının Değişimi", fontsize=12, fontweight="bold", color="#2c5282")
        ax4.set_xlabel("Maksimum Olasılık / Güven Skoru", fontsize=10)
        ax4.set_ylabel("Örnek Sayısı", fontsize=10)
        ax4.legend(loc="upper left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 5: NLL Optimizasyonu ve Optimal Sıcaklık (T*)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        t_degerleri = sicaklik_tarama["t_degerleri"]
        nll_degerleri = sicaklik_tarama["nll_degerleri"]

        ax5.plot(t_degerleri, nll_degerleri, "b-", linewidth=2.5, label="Val NLL Eğrisi")
        ax5.axvline(optimal_sicaklik, color="#e53e3e", linestyle="--", linewidth=2, label=f"Optimal T* = {optimal_sicaklik:.2f}")
        ax5.axvline(1.0, color="#718096", linestyle=":", linewidth=1.5, label="Ham Model (T = 1.0)")

        ax5.set_title("5. Sıcaklık Parametresine Göre NLL Kayıp Yüzeyi", fontsize=12, fontweight="bold", color="#553c9a")
        ax5.set_xlabel("Sıcaklık (Temperature T)", fontsize=10)
        ax5.set_ylabel("Negative Log-Likelihood (NLL)", fontsize=10)
        ax5.legend(loc="upper right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       PROBABILITY CALIBRATION (ECE & TS) SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Modelin doğruluğunu (Accuracy) ve tahminlerini ASLA değiştirmez.\n"
            "  • Sadece tek bir parametre (T*) öğrenir; saniyeler içinde eğitilir.\n"
            "  • ECE ve NLL değerlerini dramatik şekilde düşürür.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Sınıf bazında farklı aşırı güven varsa tek skaler T* yetersiz kalabilir.\n"
            "  • Ayrı bir doğrulama (validation) kümesi ayrılmasını gerektirir.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • OOD tespiti ve seçici tahmin (Abstention) mekanizmalarının temeli.\n"
            "  • Tıbbi ve otonom sistemlerde güvenilir belirsizlik modellemesi.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Dağılım kayması (Domain shift) durumunda T* kalibrasyonu bozulabilir."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Kalibrasyon SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
