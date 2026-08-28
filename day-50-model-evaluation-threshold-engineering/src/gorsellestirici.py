"""
6-Panelli Eşik Değeri Mühendisliği ve Maliyet-Fayda Teşhis Panosu (Threshold Engineering Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EsikMuhendisligiGorsellestirici:
    """Olasılık kalibrasyonu, F-Beta optimizasyonu ve net finansal kazanç eğrilerini 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        kalibrasyon_sonuc: Dict[str, Any],
        esik_sonuc: Dict[str, Any],
        hedef_path: str = "ciktilar/esik_muhendisligi_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 50: Model Değerlendirme & Eşik Değeri Mühendisliği (F-Beta, Maliyet-Fayda ve Kalibrasyon)",
            fontsize=15, fontweight="bold", y=0.98
        )

        esikler = esik_sonuc["esikler"]
        opt_f1 = esik_sonuc["optimal_f1_esigi"]
        opt_f2 = esik_sonuc["optimal_f2_esigi"]
        opt_fin = esik_sonuc["optimal_finansal_esik"]
        brier = kalibrasyon_sonuc["brier_skoru"]
        ece = kalibrasyon_sonuc["ece_skoru"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Eşik ve Maliyet Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"EŞİK MÜHENDİSLİĞİ YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Brier Skoru       : {brier:.4f} (Kalibrasyon: {kalibrasyon_sonuc['kalibrasyon_durumu']})\n"
            f"• ECE Hatası        : %{ece * 100:.2f}\n"
            f"• F1-Optimal Eşik   : {opt_f1:.3f} (Dengeli Tercih)\n"
            f"• F2-Optimal Eşik   : {opt_f2:.3f} (Recall/Risk Öncelikli)\n"
            f"• Finansal Eşik     : {opt_fin:.3f} (Maks. Net Kazanç)\n"
            f"• Maks. Net Kazanç  : ${esik_sonuc['maksimum_net_kazanc']:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Üretim Stratejisi : FİNANSAL OPTİMİZE EDİLDİ (ONAYLANDI)"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#2ecc71", alpha=0.22, edgecolor="#27ae60", linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Eşik Değeri Mühendisliği Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Güvenilirlik / Kalibrasyon Eğrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        p_true = kalibrasyon_sonuc["prob_true"]
        p_pred = kalibrasyon_sonuc["prob_pred"]

        ax2.plot(p_pred, p_true, marker="s", color="#8e44ad", linewidth=2.2, label=f"Model (ECE=%{ece*100:.2f})")
        ax2.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", label="Mükemmel Kalibre Çizgisi")

        ax2.set_title("2. Güvenilirlik & Kalibrasyon Eğrisi (Reliability Curve)", fontweight="bold", color="#8e44ad")
        ax2.set_xlabel("Ortalama Tahmin Edilen Olasılık")
        ax2.set_ylabel("Gerçekleşen Pozitif Oranı")
        ax2.legend(loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: F-Beta Skor Eğrileri (Beta = 0.5, 1.0, 2.0)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(esikler, esik_sonuc["f05_skorlari"], label="F0.5 (Precision Öncelikli)", color="#3498db", linewidth=1.8)
        ax3.plot(esikler, esik_sonuc["f1_skorlari"], label="F1.0 (Dengeli)", color="#2ecc71", linewidth=2.2)
        ax3.plot(esikler, esik_sonuc["f2_skorlari"], label="F2.0 (Recall Öncelikli)", color="#e74c3c", linewidth=1.8)

        ax3.axvline(opt_f1, color="#2ecc71", linestyle=":", label=f"F1* ({opt_f1:.2f})")
        ax3.axvline(opt_f2, color="#e74c3c", linestyle=":", label=f"F2* ({opt_f2:.2f})")

        ax3.set_title("3. F-Beta Skorları vs Karar Eşiği (Threshold)", fontweight="bold", color="#2980b9")
        ax3.set_xlabel("Karar Eşiği (Tau)")
        ax3.set_ylabel("F-Beta Skoru")
        ax3.legend(loc="lower center", fontsize=8)

        # -------------------------------------------------------------
        # Panel 4: Maliyet-Fayda Net Finansal Kazanç Eğrisi ($)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        kazanc = esik_sonuc["net_kazanc_listesi"]
        ax4.plot(esikler, kazanc, color="#27ae60", linewidth=2.4, label="Net Finansal Kazanç ($)")
        ax4.axvline(opt_fin, color="#e74c3c", linestyle="--", label=f"Optimum Eşik ({opt_fin:.2f})")
        ax4.scatter([opt_fin], [esik_sonuc["maksimum_net_kazanc"]], color="#e74c3c", s=70, zorder=5)

        ax4.set_title("4. Maliyet-Fayda Net Finansal Kazanç Eğrisi ($)", fontweight="bold", color="#27ae60")
        ax4.set_xlabel("Karar Eşiği (Tau)")
        ax4.set_ylabel("Net Kazanç ($)")
        ax4.legend(loc="lower center", fontsize=8)

        # -------------------------------------------------------------
        # Panel 5: Karar Eğrisi Analizi (Decision Curve Analysis - DCA)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        dca_nb = esik_sonuc["dca_net_benefit"]
        ax5.plot(esikler[:85], dca_nb[:85], color="#d35400", linewidth=2.2, label="Model Net Faydası (NB)")
        ax5.axhline(0, color="#7f8c8d", linestyle="--", label="Müdahale Yok (Treat None = 0)")

        ax5.set_title("5. Karar Eğrisi Analizi (Decision Curve Analysis - DCA)", fontweight="bold", color="#d35400")
        ax5.set_xlabel("Eşik Olasılığı (Risk Eşiği)")
        ax5.set_ylabel("Net Fayda (Net Benefit)")
        ax5.set_ylim(-0.05, max(dca_nb[:85]) * 1.25)
        ax5.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 6: Eşik Değişiminde Karmaşıklık Hücre Dağılımı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.plot(esikler, esik_sonuc["tp"], label="Doğru Pozitif (TP)", color="#2ecc71", linewidth=2.0)
        ax6.plot(esikler, esik_sonuc["fn"], label="Kaçırılan Risk (FN)", color="#e74c3c", linewidth=2.0)
        ax6.plot(esikler, esik_sonuc["fp"], label="Yanlış Alarm (FP)", color="#f39c12", linewidth=2.0, linestyle="--")

        ax6.axvline(opt_fin, color="#27ae60", linestyle=":", label=f"Finansal Eşik ({opt_fin:.2f})")
        ax6.set_title("6. Karmaşıklık Hücreleri vs Eşik Seyri", fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Karar Eşiği (Tau)")
        ax6.set_ylabel("Örneklem Sayısı")
        ax6.legend(loc="center right", fontsize=8)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
