"""
6-Panelli CIELAB Kolorimetri ve Delta-E 2000 Tolerans Teşhis Panosu (Palette & Tolerance Dashboard).
"""

from typing import Dict, Any, List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .delta_e_hesaplayici import DeltaEHesaplayici


class PaletAnalizGorsellestirici:
    """CIELAB renk analizi, dominant paletler ve CIEDE2000 tolerans sonuçlarını 6 panelli panoda sunar."""

    @classmethod
    def panel_ciz(
        cls,
        hedef_gorsel: np.ndarray,
        numune_gorsel: np.ndarray,
        hedef_analiz: Dict[str, Any],
        numune_analiz: Dict[str, Any],
        karsilastirma_sonuclari: List[Dict[str, Any]],
        ortalama_de00: float,
        tolerans_ozet: Dict[str, Any],
        hedef_path: str = "ciktilar/cielab_palet_tolerans_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 53: CIELAB Renk Uzayında K-Means & Delta-E 2000 Hassas Tolerans Analizi (Industrial Colorimetry)",
            fontsize=15, fontweight="bold", y=0.98
        )

        kart_renk = tolerans_ozet["renk"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Kolorimetri ve Tolerans Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"KOLORİMETRİ & KALİTE YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• K-Means Küme Sayısı (K) : {hedef_analiz['k_renk']} Dominant Renk\n"
            f"• Renk Uzayı Modeli       : Standart CIELAB D65\n"
            f"• Ortalama Delta-E 2000   : {ortalama_de00:.2f} dE00\n"
            f"• Algısal Tolerans Durumu : {tolerans_ozet['kod']}\n"
            f"• Kalite Kararı (Seviye)  : {tolerans_ozet['seviye']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Teşhis Değerlendirmesi  :\n"
            f"  {tolerans_ozet['aciklama']}"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=kart_renk, alpha=0.22, edgecolor=kart_renk, linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Kolorimetri & Kalite Teftiş Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Orijinal Referans ve Üretim Numunesi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.axis("off")

        # İki görseli yan yana birleştirme
        h_min = min(hedef_gorsel.shape[0], numune_gorsel.shape[0])
        w_min = min(hedef_gorsel.shape[1], numune_gorsel.shape[1])
        g1 = hedef_gorsel[:h_min, :w_min]
        g2 = numune_gorsel[:h_min, :w_min]
        birlesik = np.hstack([g1, g2])

        ax2.imshow(birlesik)
        ax2.set_title("2. Referans Standart (Sol) vs Üretim Partisi (Sağ)", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 3: Çıkarılan Dominant Renk Paleti ve LAB/HEX
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        palet = hedef_analiz["palet"]
        y_pos = np.arange(len(palet))

        for idx, p in enumerate(palet):
            ax3.barh(idx, p["yuzde"], color=p["hex"], edgecolor="black", height=0.6)
            ax3.text(
                p["yuzde"] + 1.0, idx,
                f"{p['hex']} | L*:{p['lab'][0]:.0f} a*:{p['lab'][1]:.0f} b*:{p['lab'][2]:.0f} (%{p['yuzde']:.1f})",
                va="center", fontsize=8.2, fontweight="bold", color="#2c3e50"
            )

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"Renk #{p['sira']}" for p in palet])
        ax3.invert_yaxis()
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("Piksel Baskınlık Oranı (%)")
        ax3.set_title("3. CIELAB K-Means Dominant Palet", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 4: a* vs b* Kromatiklik Düzlemi Dağılımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        lab_pikseller = hedef_analiz["lab_pikseller"]
        # Hızlı görselleştirme için rastgele 2500 piksel alt-örnekleme
        if len(lab_pikseller) > 2500:
            indeksler = np.random.choice(len(lab_pikseller), 2500, replace=False)
            p_ornek = lab_pikseller[indeksler]
        else:
            p_ornek = lab_pikseller

        ax4.scatter(p_ornek[:, 1], p_ornek[:, 2], alpha=0.25, c="#95a5a6", s=8, label="Pikseller")

        # Küme Merkezleri
        for idx, p in enumerate(palet):
            ax4.scatter(
                p["lab"][1], p["lab"][2], color=p["hex"], edgecolor="black",
                s=180, zorder=5, label=f"Merkez #{p['sira']}"
            )

        ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax4.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax4.set_xlabel("a* (Yeşil <-> Kırmızı)")
        ax4.set_ylabel("b* (Mavi <-> Sarı)")
        ax4.set_title("4. a* vs b* Kromatiklik Düzlemi", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 5: Delta-E 76 vs CIEDE2000 Renk Sapması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        etiketler_cift = [f"Renk #{c['sira']}" for c in karsilastirma_sonuclari]
        de76_degerler = [c["delta_e_76"] for c in karsilastirma_sonuclari]
        de00_degerler = [c["delta_e_2000"] for c in karsilastirma_sonuclari]

        x = np.arange(len(etiketler_cift))
        width = 0.35

        ax5.bar(x - width/2, de76_degerler, width, label="Delta-E 1976 (Öklid)", color="#bdc3c7", edgecolor="black")
        ax5.bar(x + width/2, de00_degerler, width, label="CIEDE2000 (Algısal)", color="#e74c3c", edgecolor="black")

        ax5.axhline(1.0, color="#2ecc71", linestyle="--", linewidth=1.5, label="dE00 < 1.0 (Kusursuz)")
        ax5.axhline(2.0, color="#f39c12", linestyle="--", linewidth=1.5, label="dE00 = 2.0 (Tolerans Sınırı)")

        ax5.set_xticks(x)
        ax5.set_xticklabels(etiketler_cift)
        ax5.set_ylabel("Renk Farkı (Delta-E)")
        ax5.set_title("5. Delta-E 76 vs CIEDE2000 Sapma Analizi", fontweight="bold", color="#c0392b")
        ax5.legend(loc="upper right", fontsize=7.5)

        # -------------------------------------------------------------
        # Panel 6: CIEDE2000 Endüstriyel Tolerans Karar Çubuğu
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        kategoriler = ["Kusursuz\n(<1.0)", "Tolerans İçi\n(1.0-2.0)", "Kabul Sınırı\n(2.0-5.0)", "Kritik Red\n(>=5.0)"]
        renk_skalasi = ["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"]
        y_skor = [1.0, 2.0, 5.0, 8.0]

        ax6.barh(kategoriler, y_skor, color=renk_skalasi, alpha=0.35, edgecolor="black", height=0.55)
        ax6.axvline(ortalama_de00, color="#e74c3c", linewidth=3, linestyle="-", label=f"Ölçülen: {ortalama_de00:.2f} dE00")

        ax6.set_xlabel("CIEDE2000 Skoru")
        ax6.set_title("6. Endüstriyel Tolerans ve Kabul Eşikleri", fontweight="bold", color="#27ae60")
        ax6.legend(loc="lower right", fontsize=8.5)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
