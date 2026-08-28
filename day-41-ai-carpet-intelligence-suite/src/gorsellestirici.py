"""
Uçtan Uca Çoklu Görev Halı Zekası 6-Panelli Fabrika Kontrol Paneli (Executive AI Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image


class HaliZekaPaketiGorsellestirici:
    """4 yapay zeka motorunun (Renk, Arama, Kusur, RAG) çıktılarını tek bir yönetim panosunda birleştirir."""

    @classmethod
    def konsolide_panel_ciz(
        cls,
        test_gorseli: Image.Image,
        teftis_raporu: Dict[str, Any],
        hedef_path: str = "ciktilar/hali_zeka_paketi_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 41: Uçtan Uca Çoklu Görev Halı Zekası Paketi (AI Carpet Intelligence Suite) Fabrika Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        kusur_res = teftis_raporu["kusur_tespiti"]
        renk_res = teftis_raporu["renk_analizi"]
        arama_res = teftis_raporu["gorsel_arama"]
        rag_res = teftis_raporu["rag_cozum_onerileri"]

        # -------------------------------------------------------------
        # Panel 1: Canlı Görsel & Kusur Bounding Box
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.imshow(test_gorseli)
        ax1.set_title("1. Dokuma Kontrol Hattı (Kusur Tespiti)", fontweight="bold", color="#1f77b4")
        ax1.axis("off")

        for k in kusur_res.get("kusurlar", []):
            x, y, w, h = k["kutu"]
            tur = k["kusur_turu"]
            c = "#e74c3c" if k["siddet"] == "KRITIK" else "#f39c12"
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=c, facecolor="none")
            ax1.add_patch(rect)
            ax1.text(
                x, max(0, y - 5), f"{tur}\n({k['siddet']})",
                color="white", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.85, edgecolor="black")
            )

        # -------------------------------------------------------------
        # Panel 2: İplik Renk Sarfiyatı & Delta-E Uyum Swatch'ları
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        iplikler = renk_res.get("iplikler", [])
        n_iplik = len(iplikler)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, max(n_iplik * 2, 2))
        ax2.axis("off")

        for idx, ip in enumerate(iplikler[::-1]):
            y = idx * 2
            k_rgb = [c / 255.0 for c in ip["katalog_rgb"]]
            ax2.add_patch(plt.Rectangle((0.5, y + 0.3), 1.8, 1.4, facecolor=k_rgb, edgecolor="black", linewidth=1.2))
            ax2.text(2.6, y + 1.0, f"{ip['iplik_id']} (%{ip['yuzde']:.1f}) -> {ip['katalog_ad']}", fontweight="bold", fontsize=8, va="center")
            dE_text = f"dE: {ip['delta_e_2000']:.2f} ({ip['uyum_durumu']})"
            ax2.text(7.6, y + 1.0, dE_text, fontweight="bold", fontsize=7.5, color="#27ae60" if ip["delta_e_2000"] < 2.0 else "#e67e22", va="center")

        ax2.set_title("2. İplik Renk Sarfiyatı & Delta-E 2000 Kartelası", fontweight="bold", color="#2ca02c")

        # -------------------------------------------------------------
        # Panel 3: Görsel Arama Katalog Eşleşmesi (Top Match)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        top_match = arama_res.get("en_iyi_eslesme")
        if top_match:
            ax3.imshow(top_match["gorsel"])
            ax3.set_title(f"3. Katalog Eşleşmesi: {top_match['ad']}\n(Benzerlik: %{top_match['benzerlik_skoru']:.1f})", fontweight="bold", color="#8e44ad", fontsize=9.5)
        else:
            ax3.text(0.5, 0.5, "Katalog Eşleşmesi Yok", ha="center", va="center")
        ax3.axis("off")

        # -------------------------------------------------------------
        # Panel 4: Pikselsel Anomali Isı Haritası
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        anomali = kusur_res.get("anomali_haritasi", np.zeros((10, 10)))
        im4 = ax4.imshow(anomali, cmap="hot", interpolation="nearest")
        ax4.set_title("4. Anomali & Kalıntı Isı Haritası", fontweight="bold", color="#d35400")
        ax4.axis("off")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # -------------------------------------------------------------
        # Panel 5: Otomatik Sektörel RAG Çözüm ve Standart Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")

        rag_metin = "OTOMATİK TEKNİK ÇÖZÜM REÇETELERİ (RAG):\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        if rag_res:
            for r in rag_res[:2]:
                rag_metin += (
                    f"• {r['kusur_turu']} ({r['siddet']}):\n"
                    f"  Standart: {r['standart']}\n"
                    f"  Reçete  : {r['oneri']}\n\n"
                )
        else:
            rag_metin += "Kusursuz Numune: Herhangi bir bakım aksiyonu gerekmemektedir."

        ax5.text(
            0.05, 0.5, rag_metin, transform=ax5.transAxes, va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdfefe", edgecolor="#7f8c8d", linewidth=1.5),
            fontsize=7.8, family="monospace"
        )
        ax5.set_title("5. Otomatik Sektörel RAG Çözüm Danışmanı", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 6: Yönetici Kalite Kararı ve KPI Özeti
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        skor = teftis_raporu["genel_kalite_skoru"]
        karar = teftis_raporu["fabrika_karari"]
        k_bg = "#2ecc71" if "1_KALITE" in karar else "#f39c12" if "2_KALITE" in karar else "#e74c3c"

        yonetici_karti = (
            f"FABRİKA GENEL TEFTİŞ KARARI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Genel Kalite Skoru: %{skor:.1f} / 100\n"
            f"• Hat Durumu        : {karar}\n"
            f"• Tespit Edilen Hata: {kusur_res.get('kusur_sayisi', 0)} Adet\n"
            f"• İplik Renk Uyumu  : {'UYGUN (PASS)' if renk_res.get('parti_renk_uyumu') else 'SAPMA VAR'}\n"
            f"• Katalog Eşleşmesi : %{top_match['benzerlik_skoru'] if top_match else 0:.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Sevkiyat Onayı    : {'ONAYLANDI (PASS)' if teftis_raporu['sevkiyat_onayi'] else 'REDDEDİLDİ (FAIL)'}"
        )

        ax6.text(
            0.5, 0.5, yonetici_karti, transform=ax6.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor=k_bg, alpha=0.25, edgecolor=k_bg, linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax6.set_title("6. Fabrika Yönetim ve Sevkiyat Kararı", fontweight="bold", color="#2c3e50")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
