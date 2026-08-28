"""
Halı Dokuma Hataları ve Kalite Kontrol 6-Panelli Teşhis Panosu.
"""

from typing import List, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image


class HaliKusurGorsellestirici:
    """Halı kusur tespit adımlarını ve fabrika kalite raporunu görselleştiren teşhis panosu."""

    @classmethod
    def kusur_paneli_ciz(
        cls,
        test_gorseli: Image.Image,
        anomali_haritasi: np.ndarray,
        temiz_maske: np.ndarray,
        tespit_edilen_kusurlar: List[Dict[str, Any]],
        parti_raporu: Dict[str, Any],
        hedef_path: str = "ciktilar/hali_kusur_tespit_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=300)
        fig.suptitle(
            "Day 39: Halı Dokuma Hataları, Leke ve Kusur Tespiti & Otomatik Kalite Kontrol Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        renk_haritasi = {
            "IPLIK_KOPMASI": "#e74c3c",       # Kırmızı
            "YAG_BOYA_LEKESI": "#8e44ad",    # Mor
            "DUGUM_TOPAKLANMA": "#f39c12",   # Turuncu
            "DELIK_YIRTIK": "#c0392b"        # Koyu Kırmızı
        }

        # -------------------------------------------------------------
        # Panel 1: Giriş Halısı ve Tespit Kutuları
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.imshow(test_gorseli)
        ax1.set_title("1. Giriş Halı Görseli & Kusur Bounding Box", fontweight="bold", color="#1f77b4")
        ax1.axis("off")

        for k in tespit_edilen_kusurlar:
            x, y, w, h = k["kutu"]
            tur = k["kusur_turu"]
            c = renk_haritasi.get(tur, "#e74c3c")

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=c, facecolor="none")
            ax1.add_patch(rect)
            ax1.text(
                x, max(0, y - 5), f"{k['kusur_id']}: {tur}\n({k['siddet']})",
                color="white", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.85, edgecolor="black")
            )

        # -------------------------------------------------------------
        # Panel 2: Anomali Kalıntı Yoğunluk Haritası
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        im2 = ax2.imshow(anomali_haritasi, cmap="hot", interpolation="nearest")
        ax2.set_title("2. Pikselsel Anomali / Kalıntı Isı Haritası", fontweight="bold", color="#d35400")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # -------------------------------------------------------------
        # Panel 3: Morfolojik Olarak Temizlenmiş İkili Maske
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.imshow(temiz_maske, cmap="gray")
        ax3.set_title("3. Morfolojik Açma/Kapama İkili Maskesi", fontweight="bold", color="#27ae60")
        ax3.axis("off")

        # -------------------------------------------------------------
        # Panel 4: Geometrik Şekil Analizi (AR vs Dairesellik)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        if tespit_edilen_kusurlar:
            for k in tespit_edilen_kusurlar:
                tur = k["kusur_turu"]
                c = renk_haritasi.get(tur, "#e74c3c")
                ax4.scatter(
                    k["en_boy_orani"], k["dairesellik"],
                    s=k["alan"] / 2.0 + 50, color=c, edgecolor="black", linewidth=1.5,
                    label=f"{k['kusur_id']} ({tur})"
                )
                ax4.annotate(k["kusur_id"], (k["en_boy_orani"], k["dairesellik"]),
                             xytext=(5, 5), textcoords="offset points", fontweight="bold", fontsize=8)

            ax4.axvline(3.2, color="red", linestyle="--", alpha=0.6, label="İplik Kopma Eşiği (AR >= 3.2)")
            ax4.axhline(0.45, color="purple", linestyle="--", alpha=0.6, label="Leke Dairesellik Eşiği (C >= 0.45)")
            ax4.set_xlabel("En-Boy Oranı (Aspect Ratio)", fontweight="bold", fontsize=9)
            ax4.set_ylabel("Dairesellik (Circularity)", fontweight="bold", fontsize=9)
            ax4.set_title("4. Kusur Geometrisi Morfolojik Ayrışımı", fontweight="bold", color="#8e44ad")
            ax4.legend(fontsize=7, loc="upper right")
        else:
            ax4.text(0.5, 0.5, "Kusur Tespit Edilmedi", ha="center", va="center")

        # -------------------------------------------------------------
        # Panel 5: Kusur Alan Sarfiyat Dağılımı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        if tespit_edilen_kusurlar:
            ids = [k["kusur_id"] for k in tespit_edilen_kusurlar]
            alanlar = [k["alan"] for k in tespit_edilen_kusurlar]
            bar_colors = [renk_haritasi.get(k["kusur_turu"], "#e74c3c") for k in tespit_edilen_kusurlar]

            bars = ax5.bar(ids, alanlar, color=bar_colors, edgecolor="black", width=0.45)
            ax5.set_ylabel("Kusur Alanı (Piksel Sayısı)", fontweight="bold", fontsize=9)
            ax5.set_title("5. Kusur Büyüklük & Alan Dağılımı", fontweight="bold", color="#2c3e50")

            for bar in bars:
                h = bar.get_height()
                ax5.annotate(f"{h} px", (bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
        else:
            ax5.text(0.5, 0.5, "Kusursuz Numune", ha="center", va="center")

        # -------------------------------------------------------------
        # Panel 6: Fabrika Kalite Kontrol Kararı & Protokolü
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        karar = parti_raporu["parti_kalite_karari"]
        k_bg = "#2ecc71" if "1_KALITE" in karar else "#f39c12" if "2_KALITE" in karar else "#e74c3c"

        info_box = (
            f"FABRİKA KALİTE KONTROL KARARI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Parti Durumu      : {karar}\n"
            f"• Toplam Kusur Sayısı: {parti_raporu['toplam_kusur_sayisi']} Adet\n"
            f"• Kritik Kusur       : {parti_raporu['kritik_kusur_sayisi']} Adet\n"
            f"• Orta Şiddet Kusur  : {parti_raporu['orta_kusur_sayisi']} Adet\n"
            f"• Küçük Kusur        : {parti_raporu['kucuk_kusur_sayisi']} Adet\n"
            f"• Kusurlu Alan       : {parti_raporu['toplam_kusurlu_alan_piksel']} px\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Üretim Hattı Kararı: {'ONAYLANDI (PASS)' if parti_raporu['parti_onayi'] else 'REDDEDİLDİ (FAIL)'}"
        )

        ax6.text(
            0.5, 0.5, info_box, transform=ax6.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor=k_bg, alpha=0.25, edgecolor=k_bg, linewidth=2),
            fontsize=9.5, fontweight="bold", family="monospace"
        )
        ax6.set_title("6. Fabrika Kalite Kontrol Özeti", fontweight="bold", color="#333333")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
