"""
6-Panelli İstatistiksel Veri Kayması ve MLOps Teşhis Panosu (Data Drift Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class VeriKaymasiGorsellestirici:
    """Veri kayması ve optimal taşıma metriklerini 6 panelli teşhis panosunda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        odak_oznitelik: str,
        referans_dizi: np.ndarray,
        uretim_dizi: np.ndarray,
        genel_rapor: Dict[str, Any],
        hedef_path: str = "ciktilar/veri_kaymasi_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 43: Veri Kayması (Data Drift) Tespiti, KS-Testi ve Wasserstein Mesafesi Teşhis Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        genel_durum = genel_rapor["genel_durum"]
        oz_rapor = genel_rapor["oznitelikler"][odak_oznitelik]
        grafik = oz_rapor["grafik_verisi"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Teşhis Kartı ve MLOps Aksiyonu
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        bg_c = "#2ecc71" if genel_durum == "DAGILIMLAR_KARARLI_NORMAL" else "#f39c12" if genel_durum == "ORTA_DUZEY_KAYMA_UYARISI" else "#e74c3c"
        kart_metni = (
            f"VERİ KAYMASI DURUMU: {genel_durum}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• MLOps Aksiyonu    : {genel_rapor['mlops_aksiyonu']}\n"
            f"• Alarm Durumu      : {'TETİKLENDİ (ALERT)' if genel_rapor['alarm_verildi'] else 'NORMAL'}\n"
            f"• Kayan Öznitelik   : {genel_rapor['kayan_oznitelik_sayisi']} / {genel_rapor['toplam_oznitelik_sayisi']} (%{genel_rapor['kayma_orani']:.1f})\n"
            f"• Kritik Drift      : {genel_rapor['kritik_kayma_sayisi']} Adet\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Odak Öznitelik    : {odak_oznitelik}\n"
            f"  - KS İstatistiği  : {oz_rapor['ks_istatistigi']:.4f}\n"
            f"  - p-değeri        : {oz_rapor['p_degeri']:.6f} (alpha={oz_rapor['alpha_esigi']})\n"
            f"  - Wasserstein W1  : {oz_rapor['wasserstein_mesafesi']:.4f}\n"
            f"  - PSI Skoru       : {oz_rapor['psi_skoru']:.4f}"
        )
        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=bg_c, alpha=0.25, edgecolor=bg_c, linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Yönetici MLOps Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Kolmogorov-Smirnov Ampirik CDF Eğrisi & Maksimum D
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        x_izgara = grafik["izgara"]
        ax2.plot(x_izgara, grafik["cdf_ref"], label="Referans CDF", color="#2980b9", linewidth=2.2)
        ax2.plot(x_izgara, grafik["cdf_prod"], label="Üretim CDF", color="#e74c3c", linewidth=2.2, linestyle="--")

        # Maksimum D mesafesini çiz
        maks_x = grafik["maks_fark_x"]
        idx = np.argmin(np.abs(x_izgara - maks_x))
        y1 = grafik["cdf_ref"][idx]
        y2 = grafik["cdf_prod"][idx]
        ax2.vlines(maks_x, min(y1, y2), max(y1, y2), color="#8e44ad", linewidth=2.5, linestyle=":", label=f"Maks D_KS = {grafik['ks_stat']:.3f}")
        ax2.scatter([maks_x, maks_x], [y1, y2], color="#8e44ad", s=30, zorder=5)

        ax2.set_title("2. Kolmogorov-Smirnov Ampirik CDF & D_KS", fontweight="bold", color="#1f77b4")
        ax2.set_xlabel("Değer Aralığı")
        ax2.set_ylabel("Kümülatif Olasılık F(x)")
        ax2.legend(loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: Wasserstein Mesafesi (Earth Mover's Taşıma Alanı)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(x_izgara, grafik["cdf_ref"], color="#2980b9", linewidth=1.5)
        ax3.plot(x_izgara, grafik["cdf_prod"], color="#e74c3c", linewidth=1.5, linestyle="--")
        ax3.fill_between(x_izgara, grafik["cdf_ref"], grafik["cdf_prod"], color="#f39c12", alpha=0.35, label=f"W1 Taşıma Alanı ({oz_rapor['wasserstein_mesafesi']:.3f})")

        ax3.set_title("3. Wasserstein (EMD) Optimal Taşıma Maliyeti", fontweight="bold", color="#d35400")
        ax3.set_xlabel("Değer Aralığı")
        ax3.set_ylabel("F_ref(x) - F_prod(x) Alanı")
        ax3.legend(loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 4: Özellik Bazlı p-Değeri ve Anlamlılık Eşiği
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        oz_adlari = list(genel_rapor["oznitelikler"].keys())
        p_degerleri = [genel_rapor["oznitelikler"][k]["p_degeri"] for k in oz_adlari]
        renkler_p = ["#2ecc71" if p >= 0.05 else "#e74c3c" for p in p_degerleri]

        b_p = ax4.bar(oz_adlari, p_degerleri, color=renkler_p, edgecolor="black", alpha=0.85)
        ax4.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Anlamlılık Eşiği (alpha=0.05)")
        for rect in b_p:
            h = rect.get_height()
            ax4.text(rect.get_x() + rect.get_width() / 2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax4.set_title("4. Özellik Bazlı KS-Testi p-Değerleri", fontweight="bold", color="#8e44ad")
        ax4.set_ylabel("p-değeri (p < 0.05 ise Drift)")
        ax4.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 5: Dağılım Karşılaştırma Histogram / KDE
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        sns.histplot(referans_dizi, bins=25, kde=True, color="#2980b9", label="Referans (Eğitim)", ax=ax5, stat="density", alpha=0.4)
        sns.histplot(uretim_dizi, bins=25, kde=True, color="#e74c3c", label="Üretim (Canlı)", ax=ax5, stat="density", alpha=0.4)

        ax5.set_title(f"5. Olasılık Yoğunluk Dağılımı ({odak_oznitelik})", fontweight="bold", color="#16a085")
        ax5.set_xlabel("Öznitelik Değeri")
        ax5.set_ylabel("Olasılık Yoğunluğu")
        ax5.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 6: Nüfus Kararlılık İndeksi (PSI) Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        psi_degerleri = [genel_rapor["oznitelikler"][k]["psi_skoru"] for k in oz_adlari]
        renkler_psi = ["#2ecc71" if p < 0.1 else "#f39c12" if p < 0.2 else "#e74c3c" for p in psi_degerleri]

        b_psi = ax6.bar(oz_adlari, psi_degerleri, color=renkler_psi, edgecolor="black", alpha=0.85)
        ax6.axhline(0.10, color="#f39c12", linestyle=":", label="Uyarı Eşiği (0.10)")
        ax6.axhline(0.20, color="#e74c3c", linestyle="--", label="Kritik Alarm Eşiği (0.20)")

        for rect in b_psi:
            h = rect.get_height()
            ax6.text(rect.get_x() + rect.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax6.set_title("6. Nüfus Kararlılık İndeksi (PSI) Seviyeleri", fontweight="bold", color="#c0392b")
        ax6.set_ylabel("PSI Skoru")
        ax6.legend(loc="upper right", fontsize=8)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
