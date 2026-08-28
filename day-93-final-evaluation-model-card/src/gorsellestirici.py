"""
Day 93: 6-Panelli Kapsamlı Değerlendirme, Yanlılık ve Model Card Teşhis Panosu
-----------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class DegerlendirmeGorsellestirici:
    """
    Model performansını, sınıf karışıklık matrisini, dilim performanslarını,
    adillik metriklerini ve kalibrasyon eğrisini 6 panelli profesyonel panoda görselleştirir.
    """

    def __init__(self, cizim_boyutu: tuple = (18, 12), dpi: int = 300):
        self.cizim_boyutu = cizim_boyutu
        self.dpi = dpi

    def olustur_degerlendirme_paneli(
        self,
        metrikler: Any,
        adillik_raporu: Any,
        metadata: Any,
        kayit_yolu: str,
    ) -> None:
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, eksenler = plt.subplots(2, 3, figsize=self.cizim_boyutu, dpi=self.dpi)
        fig.suptitle(
            "Day 93: Kapsamli Model Degerlendirme, Yanlilik (Bias) ve Model Card Panosu",
            fontsize=16,
            fontweight="bold",
            color="#111827",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Genel Performans Metrikleri
        # -------------------------------------------------------------
        ax1 = eksenler[0, 0]
        metrik_adlari = ["Accuracy", "Macro F1", "Weighted F1", "Precision", "Recall"]
        metrik_degerleri = [
            metrikler.dogruluk,
            metrikler.macro_f1,
            metrikler.weighted_f1,
            metrikler.macro_precision,
            metrikler.macro_recall,
        ]
        renkler1 = ["#2563eb", "#3b82f6", "#60a5fa", "#10b981", "#059669"]
        cubuklar1 = ax1.bar(metrik_adlari, metrik_degerleri, color=renkler1, alpha=0.85, edgecolor="#374151")
        ax1.set_ylim(0, 1.15)
        for c, v in zip(cubuklar1, metrik_degerleri):
            ax1.text(c.get_x() + c.get_width() / 2, v + 0.02, f"%{v * 100:.1f}", ha="center", fontweight="bold", fontsize=9)
        ax1.set_title("1. Genel Performans Metrikleri", fontsize=11, fontweight="bold", color="#1f2937")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: Karışıklık Matrisi (Confusion Matrix Heatmap)
        # -------------------------------------------------------------
        ax2 = eksenler[0, 1]
        cm = metrikler.karisiklik_matrisi
        im = ax2.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("2. Sınıf Bazlı Karışıklık Matrisi", fontsize=11, fontweight="bold", color="#1f2937")
        ax2.set_xlabel("Tahmin Edilen Sınıf", fontsize=9)
        ax2.set_ylabel("Gerçek Sınıf", fontsize=9)
        ax2.grid(False)

        # -------------------------------------------------------------
        # PANEL 3: Dilim (Slice) Bazlı Doğruluk & F1
        # -------------------------------------------------------------
        ax3 = eksenler[0, 2]
        if adillik_raporu.dilim_sonuclari:
            d_isimler = list(adillik_raporu.dilim_sonuclari.keys())
            d_accler = [adillik_raporu.dilim_sonuclari[k].dogruluk for k in d_isimler]
            d_f1ler = [adillik_raporu.dilim_sonuclari[k].f1_skoru for k in d_isimler]

            x_pos = np.arange(len(d_isimler))
            w = 0.35
            ax3.bar(x_pos - w / 2, d_accler, w, label="Doğruluk (Acc)", color="#6366f1", alpha=0.8)
            ax3.bar(x_pos + w / 2, d_f1ler, w, label="F1-Skoru", color="#ec4899", alpha=0.8)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(d_isimler, rotation=20, fontsize=8)
            ax3.set_ylim(0, 1.15)
            ax3.legend(loc="upper right", fontsize=8)
        ax3.set_title("3. Alt Grup / Dilim Performansı", fontsize=11, fontweight="bold", color="#1f2937")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Adillik ve Yanlılık Metrikleri (80% Kuralı)
        # -------------------------------------------------------------
        ax4 = eksenler[1, 0]
        adillik_metrik_adlari = ["Disparate Impact\n(Hedef >= %80)", "Demographic\nParity Farkı", "Maks Dilim\nDoğruluk Farkı"]
        adillik_metrik_degerleri = [
            adillik_raporu.disparate_impact_orani,
            adillik_raporu.demographic_parity_farki,
            adillik_raporu.maks_dogruluk_farki,
        ]
        renkler4 = ["#10b981" if adillik_raporu.disparate_impact_orani >= 0.8 else "#ef4444", "#f59e0b", "#8b5cf6"]
        cubuklar4 = ax4.bar(adillik_metrik_adlari, adillik_metrik_degerleri, color=renkler4, alpha=0.85, edgecolor="#374151")
        ax4.axhline(0.80, color="#10b981", linestyle="--", label="DIR Eşiği (%80)")
        ax4.set_ylim(0, 1.2)
        for c, v in zip(cubuklar4, adillik_metrik_degerleri):
            ax4.text(c.get_x() + c.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
        ax4.set_title("4. Adillik & Yanlılık Denetim Metrikleri", fontsize=11, fontweight="bold", color="#1f2937")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Güvenilirlik & Olasılık Kalibrasyon Eğrisi (ECE)
        # -------------------------------------------------------------
        ax5 = eksenler[1, 1]
        kalibrasyon = metrikler.kalibrasyon
        if kalibrasyon.kutu_guvenleri:
            ax5.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Mükemmel Kalibrasyon")
            ax5.plot(kalibrasyon.kutu_guvenleri, kalibrasyon.kutu_dogruluklari, marker="o", lw=2, color="#ef4444", label=f"Model (ECE: {kalibrasyon.ece_skoru:.4f})")
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.set_xlabel("Ortalama Güven Skoru (Confidence)", fontsize=9)
            ax5.set_ylabel("Gerçek Doğruluk (Accuracy)", fontsize=9)
            ax5.legend(loc="upper left", fontsize=8)
        ax5.set_title("5. Olasılık Kalibrasyonu (Reliability Diagram)", fontsize=11, fontweight="bold", color="#1f2937")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: Model Card & Etik Uyumluluk Özet Kartı
        # -------------------------------------------------------------
        ax6 = eksenler[1, 2]
        ax6.axis("off")

        durum_renk = "#10b981" if adillik_raporu.adillik_esigi_gecti_mi else "#ef4444"
        durum_metni = "ONAYLANDI (YAYINA HAZIR)" if adillik_raporu.adillik_esigi_gecti_mi else "DİKKAT (YANLILIK RİSKİ)"

        satirlar = [
            f"MODEL CARD ÖZETİ: {metadata.model_adi}",
            "─" * 40,
            f"• Sürüm              : {metadata.surum}",
            f"• Parametre Sayısı   : {metadata.parametre_sayisi:,}",
            f"• Test Doğruluğu     : %{metrikler.dogruluk * 100:.2f}",
            f"• Macro F1-Skoru     : {metrikler.macro_f1:.4f}",
            f"• ECE Kalibrasyon    : {kalibrasyon.ece_skoru:.4f}",
            f"• Brier Skoru        : {kalibrasyon.brier_skoru:.4f}",
            "─" * 40,
            f"• Disparate Impact   : %{adillik_raporu.disparate_impact_orani * 100:.1f} ({'GEÇTİ' if adillik_raporu.disparate_impact_orani >= 0.8 else 'KALDI'})",
            f"• Dilim Doğruluk Fark: %{adillik_raporu.maks_dogruluk_farki * 100:.1f}",
            f"• Etik Karar         : {durum_metni}",
        ]

        kutu_metni = "\n".join(satirlar)
        ax6.text(
            0.05,
            0.5,
            kutu_metni,
            fontsize=10,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8fafc", edgecolor=durum_renk, lw=2.5),
        )
        ax6.set_title("6. Model Card & Yayın Onay Kartı", fontsize=11, fontweight="bold", color="#1f2937")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
