"""
Tümdengelimsel Mantık ve Safsata Dedektörü Teşhis Panosu Görselleştirici Modülü (Day 153 - Faz 8).
6 panelli Geçerlilik/Sağlamlık Dağılımı, Safsata Türleri, Güven Skorları, Kıyas Tablosu, Akış Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class MantikGorsellestirici:
    """Mantıksal akıl yürütme ve safsata teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        degerlendirme_listesi: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/logical_fallacy_deductive_engine_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 153: Tümdengelimsel Mantık Doğrulayıcı & Mantıksal Safsata (Logical Fallacy) Dedektörü",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Argümanların Geçerlilik ve Sağlamlık Durumu
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        arg_isimleri = [f"Arg {i}" for i in range(1, len(degerlendirme_listesi) + 1)]
        gecerlilikler = [1 if d["gecerli_mi"] else 0 for d in degerlendirme_listesi]
        saglamliklar = [1 if d["saglam_mi"] else 0 for d in degerlendirme_listesi]

        x = np.arange(len(arg_isimleri))
        width = 0.35

        ax1.bar(x - width/2, gecerlilikler, width, label="Geçerli (Valid)", color="#4e73df", edgecolor="black")
        ax1.bar(x + width/2, saglamliklar, width, label="Sağlam (Sound)", color="#1cc88a", edgecolor="black")

        ax1.set_title("1. Argümanların Geçerlilik & Sağlamlık Analizi", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(arg_isimleri)
        ax1.set_ylabel("Durum (1: Evet, 0: Hayır)")
        ax1.set_ylim(0, 1.3)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Mantıksal Safsata Dağılımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        safsata_kategorileri = ["Geçerli (Sağlam)", "Sonucun Doğrulanması", "Kişiye Saldırı (Ad Hominem)", "Korkuluk (Straw Man)", "Yanlış İkilem"]
        sayilar = [1, 1, 1, 1, 1]
        renkler2 = ["#1cc88a", "#e74a3b", "#e67e22", "#9b59b6", "#f1c40f"]

        ax2.pie(sayilar, labels=safsata_kategorileri, autopct="%1.0f%%", colors=renkler2, startangle=140, explode=(0.08, 0.05, 0.05, 0.05, 0.05))
        ax2.set_title("2. Taranan Safsata Türleri Dağılımı", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: Argüman Güven Skorları
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        guven_skorlari = [d["guven_skoru"] * 100 for d in degerlendirme_listesi]
        renkler3 = ["#1cc88a" if s == 100 else "#e74a3b" for s in guven_skorlari]

        barlar3 = ax3.bar(arg_isimleri, guven_skorlari, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 2, f"%{h:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax3.set_title("3. Argüman Mantıksal Sağlamlık Güven Skoru", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Güven Skoru (%)")
        ax3.set_ylim(0, 120)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Kıyas Ayrıştırma ve Safsata Günlüğü
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Argüman Ayrıştırma & Teşhis Günlüğü", fontsize=12, fontweight="bold", pad=10)

        gunluk_metni = "====================================================\n"
        gunluk_metni += "        ARGÜMAN ANALİZİ VE MANTIK RAPORU            \n"
        gunluk_metni += "====================================================\n"
        for i, d in enumerate(degerlendirme_listesi, start=1):
            gunluk_metni += f"[Argüman {i}]: '{d['ham_arguman'][:45]}...'\n"
            gunluk_metni += f"  • Öncül Sayısı : {len(d['onculler'])} | Sonuç: '{d['sonuc'][:30]}...'\n"
            gunluk_metni += f"  • Geçerli mi   : {d['gecerli_mi']} | Sağlam mı: {d['saglam_mi']}\n"
            gunluk_metni += f"  • Safsata Türü : {d['safsata_bilgisi']['safsata_adi']}\n"
            gunluk_metni += "----------------------------------------------------\n"

        ax4.text(
            0.02, 0.5, gunluk_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Mantıksal Dedektör Mimarisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Tümdengelim & Safsata Dedektörü Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "      DEDUCTIVE & FALLACY ENGINE PIPELINE           \n"
            "====================================================\n"
            "  [1. Doğal Dil Argümanı Girdisi]                   \n"
            "                 │                                  \n"
            "                 ▼                                  \n"
            "  [2. Öncül-Sonuç Ayrıştırıcı (Premise-Conclusion)] \n"
            "    Öncül 1: P ==> Q | Öncül 2: P | Sonuç: Q           \n"
            "                 │                                  \n"
            "        ┌────────┴────────┐                         \n"
            "        ▼                 ▼                         \n"
            "  [3. Biçimsel Mantık]  [4. Safsata Dedektörü]      \n"
            "     (Modus Ponens/     (Ad Hominem, Straw Man,     \n"
            "      Modus Tollens)     Affirming Consequent)      \n"
            "        │                 │                         \n"
            "        └────────┬────────┘                         \n"
            "                 ▼                                  \n"
            "  [5. Sağlamlık (Soundness) & Güven Skoru Raporu]   \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 153 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 153 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "    DAY 153 SUMMARY: LOGICAL FALLACY & DEDUCTION    \n"
            "====================================================\n"
            "• Ana Modüller         : OnculSonucAyristirici, SafsataTespitcisi,\n"
            "                         TumdengelimMotoru\n"
            "• Temel Kural          : Soundness = Validity ∧ True Premises\n"
            "• Desteklenen Safsatalar: Affirming Consequent, Ad Hominem,\n"
            "                         Straw Man, False Dilemma, Circular\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Argümanları öncül ve sonuç parçalarına yapılandırma\n"
            "  2. Biçimsel (Formal) vs Gayriresmi (Informal) safsata ayrımı\n"
            "  3. LLM çıktılarında mantıksal tutarlılık denetimi\n"
            "  4. Güvenilir ve çelişkisiz akıl yürütme mimarisi\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 154 (GSM8K & MATH Aritmetik Motoru)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.8,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Mantıksal Safsata Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
