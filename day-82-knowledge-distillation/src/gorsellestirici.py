"""
Knowledge Distillation Teşhis ve Görselleştirme Panosu
------------------------------------------------------
6 panelli yüksek çözünürlüklü bilgi damıtma mimarisi, sıcaklık (Temperature τ) etkisi,
öğrenci vs öğretmen başarım karşılaştırması ve kayıp dinamikleri paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F


class DamitmaGorsellestirici:
    """
    Knowledge Distillation süreçlerini, sıcaklık eğrilerini ve model kıyaslamalarını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_damitma_paneli(
        self,
        ornek_logitler: torch.Tensor,
        bagimsiz_gecmis: Dict[str, List[float]],
        damitilmis_gecmis: Dict[str, List[float]],
        ogretmen_acc: float,
        model_parametreleri: Dict[str, int],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Knowledge Distillation teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 82: Öğretmen-Öğrenci Modeli Bilgi Damıtma (Knowledge Distillation) ve Sıcaklık (τ) Dinamikleri Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        epoklar = list(range(1, len(bagimsiz_gecmis["dogrulama_dogruluk"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: Knowledge Distillation Matematiksel Akış Şeması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        kd_metin = (
            "          KNOWLEDGE DISTILLATION (HINTON ET AL. 2015)\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. ÖĞRETMEN MODELİ (Teacher - Dondurulmuş):\n"
            "     • z_t = Teacher(x) ──> p_t^τ = Softmax(z_t / τ)\n\n"
            "  2. ÖĞRENCİ MODELİ (Student - Eğitilebilir):\n"
            "     • z_s = Student(x) ──> p_s^τ = Softmax(z_s / τ)\n\n"
            "  3. BİLEŞİK KAYIP FONKSİYONU (Combined Loss):\n"
            "     • L_KD = (1 - α) · CE(z_s, y) + α · τ² · KL(p_s^τ || p_t^τ)\n\n"
            "  4. ÇIKARIM AŞAMASI (Deployment):\n"
            "     • Öğretmen atılır, sadece ultra hafif Öğrenci modeli\n"
            "       mobil / kenar cihazlara dağıtılır (τ = 1)!"
        )
        ax1.text(
            0.5, 0.5, kd_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Knowledge Distillation Hesaplama Akışı", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Sıcaklık Katsayısının (τ) Olasılık Dağılımına Etkisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        siniflar = [f"Sınıf {i}" for i in range(len(ornek_logitler))]
        tau_listesi = [1.0, 3.0, 6.0, 12.0]
        renkler = ["#e53e3e", "#dd6b20", "#3182ce", "#805ad5"]

        genislik = 0.2
        x_ind = np.arange(len(siniflar))

        for idx, tau in enumerate(tau_listesi):
            probs = F.softmax(ornek_logitler / tau, dim=-1).detach().cpu().numpy()
            ax2.bar(x_ind + (idx - 1.5) * genislik, probs, width=genislik, label=f"τ = {tau:.0f}", color=renkler[idx])

        ax2.set_title("2. Sıcaklık (τ) ile Gizli Bilginin (Dark Knowledge) Açığa Çıkması", fontsize=12, fontweight="bold", color="#2c5282")
        ax2.set_xlabel("Sınıflar", fontsize=10)
        ax2.set_ylabel("Yumuşatılmış Olasılık Dağılımı", fontsize=10)
        ax2.set_xticks(x_ind)
        ax2.set_xticklabels(siniflar, fontsize=8)
        ax2.legend(loc="upper right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 3: Model Parametre ve Boyut Karşılaştırması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        modeller = list(model_parametreleri.keys())
        params = list(model_parametreleri.values())
        bar_colors = ["#4a5568", "#48bb78"]

        bars3 = ax3.bar(modeller, params, color=bar_colors, width=0.45, edgecolor="#2d3748")
        ax3.set_title("3. Model Parametre Kapasitesi Karşılaştırması", fontsize=12, fontweight="bold", color="#2d3748")
        ax3.set_ylabel("Toplam Parametre Sayısı", fontsize=10)
        ax3.set_ylim(0, max(params) * 1.25)

        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval + (max(params)*0.03), f"{yval:,} param", ha="center", va="bottom", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Öğrenci Doğruluk Karşılaştırması (Pure CE vs KD)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epoklar, bagimsiz_gecmis["dogrulama_dogruluk"], "r--o", linewidth=2, label="Bağımsız Öğrenci (Pure CE)")
        ax4.plot(epoklar, damitilmis_gecmis["dogrulama_dogruluk"], "g-s", linewidth=2.5, label="Damıtılmış Öğrenci (KD Student)")
        ax4.axhline(ogretmen_acc, color="#4a5568", linestyle=":", linewidth=2, label=f"Öğretmen Modeli (%{ogretmen_acc:.1f})")

        ax4.set_title("4. Öğrenci Doğruluk Kıyaslaması (Val Top-1 %)", fontsize=12, fontweight="bold", color="#22543d")
        ax4.set_xlabel("Epok", fontsize=10)
        ax4.set_ylabel("Doğruluk (%)", fontsize=10)
        ax4.legend(loc="lower right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 5: Kayıp Bileşenleri Dinamiği (Hard CE vs Soft KL)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(epoklar, damitilmis_gecmis["ce_kaybi"], "o-", color="#e53e3e", linewidth=2, label="Sert Etiket Kaybı (CE)")
        ax5.plot(epoklar, damitilmis_gecmis["kl_kaybi"], "s--", color="#3182ce", linewidth=2, label="Yumuşak Damıtma Kaybı (KL x τ²)")
        ax5.plot(epoklar, damitilmis_gecmis["toplam_kayip"], "m-", linewidth=2.2, label="Toplam Bileşik Kayıp")

        ax5.set_title("5. Damıtma Kayıp Bileşenlerinin İlerlemesi", fontsize=12, fontweight="bold", color="#553c9a")
        ax5.set_xlabel("Epok", fontsize=10)
        ax5.set_ylabel("Kayıp Değeri", fontsize=10)
        ax5.legend(loc="upper right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       KNOWLEDGE DISTILLATION (KD) SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Kompakt modeller tek başına ulaşamayacağı doğruluk seviyelerine çıkar.\n"
            "  • Karanlık bilgi (Dark Knowledge) ile zengin sınıf korelasyonları aktarılır.\n"
            "  • Çıkarım aşamasında orijinal küçük model hızında 0 ms ek gecikme.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • İki aşamalı eğitim: Önce büyük öğretmenin eğitilmesi şarttır.\n"
            "  • Sıcaklık (τ) ve Alfa (α) hiperparametre dengesi hassasiyeti.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Mobil, IoT ve Edge AI cihazlarında devasa modellerin zekasını çalıştırma.\n"
            "  • Kuantizasyon (INT8) ve Budama (Pruning) ile birleştirilebilir.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Eğer öğretmen yetersiz veya aşırı uydurmuşsa yanlış bilgi damıtılır."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Knowledge Distillation SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
