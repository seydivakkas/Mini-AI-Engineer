"""
6-Panelli Dijital Adli Bilişim ve ELA Görsel Teşhis Panosu (Forensics & ELA Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AdliTeftisGorsellestirici:
    """Error Level Analysis (ELA), sensör gürültüsü ve manipülasyon skorlarını 6 panelli panoda sunar."""

    @classmethod
    def panel_ciz(
        cls,
        teftis_sonuc: Dict[str, Any],
        hedef_path: str = "ciktilar/adli_teftis_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 54: Dijital Adli Bilişim, Error Level Analysis (ELA) ve Görsel Manipülasyon Tespiti",
            fontsize=15, fontweight="bold", y=0.98
        )

        skor = teftis_sonuc["manipulasyon_skoru"]
        karar = teftis_sonuc["karar"]
        risk = teftis_sonuc["risk_seviyesi"]
        kart_renk = teftis_sonuc["karar_renk"]
        bolgeler = teftis_sonuc["supheli_bolgeler"]
        ela_ist = teftis_sonuc["ela_istatistik"]

        # -------------------------------------------------------------
        # Panel 1: Adli Bilişim Yönetici Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"ADLİ BİLİŞİM YÖNETİCİ RAPORU\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Manipülasyon Skoru      : %{skor:.1f} / 100\n"
            f"• Adli Teftiş Kararı      : {karar}\n"
            f"• Risk Seviyesi           : {risk}\n"
            f"• ELA Ortalama Hata       : {ela_ist['ortalama_hata']:.2f} px\n"
            f"• ELA Maksimum Hata       : {ela_ist['maks_hata']:.2f} px\n"
            f"• Gürültü Tutarsızlık (CV): {teftis_sonuc['gurultu_tutarsizlik_cv']:.3f}\n"
            f"• Şüpheli Bölge Sayısı    : {len(bolgeler)} Adet (%{teftis_sonuc['supheli_alan_orani']:.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Teşhis Özeti:\n"
            f"  {teftis_sonuc['aciklama']}"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=kart_renk, alpha=0.22, edgecolor=kart_renk, linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Adli Teftiş ve Sahtecilik Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Aday Görsel ve Tespit Edilen Manipüle Edilmiş Alanlar
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.imshow(teftis_sonuc["anotasyonlu_gorsel"])
        ax2.axis("off")
        ax2.set_title(f"2. Aday Görsel & Tespit Edilen Sahtecilik ({len(bolgeler)} Bölge)", fontweight="bold", color="#c0392b")

        # -------------------------------------------------------------
        # Panel 3: Error Level Analysis (ELA) Isı Haritası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.imshow(teftis_sonuc["ela_rgb"])
        ax3.axis("off")
        ax3.set_title("3. Error Level Analysis (ELA 15x Fark Isı Haritası)", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 4: Sensör Gürültü Kalıntısı (Noise Residual)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.imshow(teftis_sonuc["kalinti_norm"], cmap="gray")
        ax4.axis("off")
        ax4.set_title("4. Sensör Gürültü Kalıntısı (PRNU Residual)", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 5: İkili Manipülasyon Maskesi (Binary Forgery Mask)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.imshow(teftis_sonuc["ikili_maske"], cmap="magma")
        ax5.axis("off")
        ax5.set_title("5. Lokal Anomali ve Sahtecilik İkili Maskesi", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # Panel 6: Sahtecilik Güven Skoru ve Risk Göstergesi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        kategoriler = ["Orijinal\n(<25)", "Şüpheli\n(25-60)", "Manipüle\n(>=60)"]
        renkler = ["#2ecc71", "#f39c12", "#e74c3c"]
        y_skor = [25.0, 60.0, 100.0]

        bars = ax6.bar(kategoriler, y_skor, color=renkler, alpha=0.35, edgecolor="black", width=0.55)
        ax6.axhline(skor, color="#e74c3c", linewidth=3, linestyle="--", label=f"Skor: %{skor:.1f}")

        ax6.set_ylabel("Manipülasyon Skoru (%)")
        ax6.set_ylim(0, 110)
        ax6.set_title("6. Adli Karar ve Risk Seviyesi Göstergesi", fontweight="bold", color="#2c3e50")
        ax6.legend(loc="upper left", fontsize=8.5)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
