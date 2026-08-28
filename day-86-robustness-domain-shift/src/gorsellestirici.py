"""
Model Dayanıklılık ve Bozulma Teşhis Panosu
-------------------------------------------
6 panelli yüksek çözünürlüklü bozulma tipleri (Noise, Blur, Weather, Digital),
şiddet seviyesi (Severity 1..5) bozulma eğrileri, mCE ve Rel-mCE metrik paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DayaniklilikGorsellestirici:
    """
    Model dayanıklılık analizlerini ve bozulma kıyaslamalarını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_dayaniklilik_paneli(
        self,
        standart_rapor: Dict[str, Any],
        dayanikli_rapor: Dict[str, Any],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Model Dayanıklılık ve Domain Shift teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 86: Görsel Bozulmalar (Bulanıklık/Gürültü) Altında Model Dayanıklılığı & Domain Shift Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Bozulmalar ve Domain Shift Akışı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        kavram_metin = (
            "      IMAGE CORRUPTIONS & DOMAIN SHIFT (ICLR 2019)\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. LABORATUVAR VS GERÇEK DÜNYA (The Robustness Gap):\n"
            "     • Laboratuvarda %95 doğru olan model; yağmur, sis veya\n"
            "       kamera gürültüsü altında %30'a kadar çökebilir!\n\n"
            "  2. 8 TEMEL BOZULMA TİPİ VE 5 ŞİDDET SEVİYESİ:\n"
            "     • Gürültü: Gauss, Tuz & Biber | Bulanıklık: Gauss, Hareket\n"
            "     • Dijital/Hava: Parlaklık, Kontrast, Pikselleme, JPEG\n\n"
            "  3. STANDART BENCHMARK METRİKLERİ:\n"
            "     • mCE (Mean Corruption Error): Tüm bozulmalardaki ort. hata\n"
            "     • Rel-mCE = mCE - Clean_Error (Salt bozulma kaybı)\n\n"
            "  4. ÇÖZÜM (Robust Augmentation): Veri uzayına pertürbasyon\n"
            "     enjekte ederek karar sınırlarını pürüzsüzleştirmek."
        )
        ax1.text(
            0.5, 0.5, kavram_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Model Dayanıklılığı ve Bozulma Mimarisi", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Bozulma Tipleri Özeti
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.axis("off")
        
        tipler_metin = (
            "         BOZULMA SINIFLARI VE MATEMATİKSEL MODELLERİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  • GAUSSIAN NOISE:    x_c = x + N(0, σ²),  σ ∈ [0.10, 0.70]\n"
            "  • SALT & PEPPER:     Rastgele pikseller ±2.5 yapılır (Impulse)\n"
            "  • GAUSSIAN BLUR:     x_c = x ⊛ G_2D(σ),   σ ∈ [0.6, 3.0]\n"
            "  • MOTION BLUR:       Yatay/çapraz hareket çekirdeği (3..11 px)\n"
            "  • BRIGHTNESS:        x_c = x + Δ,         Δ ∈ [0.15, 1.05]\n"
            "  • CONTRAST:          x_c = (x - μ) · α + μ, α ∈ [0.80, 0.20]\n"
            "  • PIXELATE:          Downsample (15%) + Nearest Upsample\n"
            "  • JPEG COMPRESSION:  Yüksek frekans kuantizasyon simülasyonu"
        )
        ax2.text(
            0.5, 0.5, tipler_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f0fff4", edgecolor="#38a169", linewidth=1.8)
        )
        ax2.set_title("2. 8 Bozulma Tipi ve Matematiksel Modelleri", fontsize=12, fontweight="bold", color="#22543d")

        # -------------------------------------------------------------
        # PANEL 3: Şiddet Seviyesine Göre Doğruluk Düşüş Eğrisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        siddetler = ["Temiz", "Şiddet 1", "Şiddet 2", "Şiddet 3", "Şiddet 4", "Şiddet 5"]
        std_egri = [standart_rapor["temiz_dogruluk"]] + list(standart_rapor["siddet_egrisi"])
        day_egri = [dayanikli_rapor["temiz_dogruluk"]] + list(dayanikli_rapor["siddet_egrisi"])

        ax3.plot(siddetler, std_egri, "r--o", linewidth=2.2, label=f"Standart Model (mAcc: %{standart_rapor['macc']:.1f})")
        ax3.plot(siddetler, day_egri, "g-s", linewidth=2.5, label=f"Dayanıklı Model (mAcc: %{dayanikli_rapor['macc']:.1f})")

        ax3.set_title("3. Bozulma Şiddetine Göre Doğruluk Düşüşü (Clean ──> s5)", fontsize=12, fontweight="bold", color="#c53030")
        ax3.set_xlabel("Bozulma Seviyesi (Severity)", fontsize=10)
        ax3.set_ylabel("Ortalama Doğruluk (%)", fontsize=10)
        ax3.set_ylim(0, 110)
        ax3.legend(loc="lower left", frameon=True)

        # -------------------------------------------------------------
        # PANEL 4: Bozulma Tiplerine Göre Doğruluk Dağılımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        bozulma_adlari = list(standart_rapor["bozulma_dogruluklari"].keys())
        std_ort_acc = [np.mean(standart_rapor["bozulma_dogruluklari"][k]) for k in bozulma_adlari]
        day_ort_acc = [np.mean(dayanikli_rapor["bozulma_dogruluklari"][k]) for k in bozulma_adlari]

        y_idx = np.arange(len(bozulma_adlari))
        h = 0.35

        ax4.barh(y_idx - h/2, std_ort_acc, height=h, color="#e53e3e", alpha=0.85, label="Standart Model")
        ax4.barh(y_idx + h/2, day_ort_acc, height=h, color="#38a169", alpha=0.85, label="Dayanıklı Model")

        ax4.set_title("4. Bozulma Tipine Göre Ortalama Doğruluk", fontsize=12, fontweight="bold", color="#2c5282")
        ax4.set_xlabel("Ortalama Doğruluk (%)", fontsize=10)
        ax4.set_yticks(y_idx)
        ax4.set_yticklabels(bozulma_adlari, fontsize=8.5)
        ax4.set_xlim(0, 110)
        ax4.legend(loc="lower right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 5: mCE ve Rel-mCE Karşılaştırması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        kats = ["mCE (Ort. Bozulma Hatası)", "Rel-mCE (Göreceli Bozulma)"]
        std_degerler = [standart_rapor["mce"], standart_rapor["rel_mce"]]
        day_degerler = [dayanikli_rapor["mce"], dayanikli_rapor["rel_mce"]]

        x_k = np.arange(len(kats))
        w = 0.35

        b1 = ax5.bar(x_k - w/2, std_degerler, width=w, color="#e53e3e", edgecolor="#2d3748", label="Standart Model")
        b2 = ax5.bar(x_k + w/2, day_degerler, width=w, color="#38a169", edgecolor="#2d3748", label="Dayanıklı Model")

        ax5.set_title("5. mCE ve Rel-mCE Hata Karşılaştırması (Düşük = Daha İyi)", fontsize=12, fontweight="bold", color="#553c9a")
        ax5.set_ylabel("Hata Oranı (%)", fontsize=10)
        ax5.set_xticks(x_k)
        ax5.set_xticklabels(kats, fontsize=9.5)
        ax5.set_ylim(0, max(std_degerler) * 1.35)
        ax5.legend(loc="upper right", frameon=True)

        for bar in b1:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"%{yval:.1f}", ha="center", fontsize=9, fontweight="bold")
        for bar in b2:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"%{yval:.1f}", ha="center", fontsize=9, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       MODEL ROBUSTNESS & DOMAIN SHIFT SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Gerçek dünya gürültüsü ve hava koşullarında istikrarlı başarım.\n"
            "  • mCE ve Rel-mCE metrikleriyle objektif dayanıklılık ölçümü.\n"
            "  • Otonom sürüş ve güvenlik kameralarında sıfır hata toleransı.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Bozulma artırma (AugMix) eğitimi ek eğitim süresi gerektirir.\n"
            "  • Aşırı agresif bozulma uygulanırsa temiz doğruluk (%1-2) düşebilir.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Endüstriyel kalite kontrol ve CCTV kameralarında güvenilirlik.\n"
            "  • OOD tespiti ve Kalibrasyon katmanlarıyla birleşerek tam güvenlik.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Eğitilmemiş yeni bozulma tiplerinde genelleme açığı oluşabilir."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Model Dayanıklılık SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
