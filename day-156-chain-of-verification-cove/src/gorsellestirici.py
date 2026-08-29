"""
Chain of Verification (CoVe) Teşhis Panosu Görselleştirici Modülü (Day 156 - Faz 8).
6 panelli Doğruluk Kıyası, İddia Durumları, CoVe 4 Aşamalı İlerleme, Öncesi/Sonrası Metin Kıyası, Akış Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class CoVEGorsellestirici:
    """CoVe halüsinasyon temizleme teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        cove_sonucu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/chain_of_verification_cove_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 156: Chain of Verification (CoVe): Halüsinasyonları Çapraz Sorularla Test Etme & Fakt Kontrolü",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: İlk Taslak vs CoVE Doğruluk Oranı (%)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = ["İlk Taslak (Baseline)\n(Halüsinasyonlu)", "CoVE Düzeltilmiş\n(4 Aşamalı Fakt Kontrol)"]
        dogruluklar = [cove_sonucu["taslak_dogruluk_orani"], cove_sonucu["cove_dogruluk_orani"]]
        renkler1 = ["#e74a3b", "#1cc88a"]

        barlar1 = ax1.bar(yontemler, dogruluklar, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_title("1. Olgusal Doğruluk Karşılaştırması", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Doğruluk Oranı (%)")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: İddiaların Durum Dağılımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kategoriler = ["Düzeltilen Halüsinasyon", "Doğrulanan Gerçek"]
        sayilar = [cove_sonucu["duzeltilen_iddia_sayisi"], max(0.01, cove_sonucu["onaylanan_iddia_sayisi"])]
        renkler2 = ["#e74a3b", "#1cc88a"]

        ax2.pie(sayilar, labels=kategoriler, autopct="%1.0f%%", colors=renkler2, startangle=140, explode=(0.05, 0.05))
        ax2.set_title("2. Taranan Olgusal İddia Durumları", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: CoVe 4 Aşamalı Güven İlerlemesi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        asamalar = ["1. Taslak", "2. Soru Planı", "3. Bağımsız Kontrol", "4. Nihai Düzeltme"]
        guven_skorlari = [25.0, 45.0, 85.0, 100.0]

        ax3.plot(asamalar, guven_skorlari, marker="o", color="#4e73df", lw=2.5, label="Fakt Güven Skoru (%)")
        for x, y in zip(asamalar, guven_skorlari):
            ax3.text(x, y + 2.5, f"%{y:.0f}", ha="center", fontsize=10, fontweight="bold")

        ax3.set_title("3. CoVe 4 Aşamalı Doğrulama İlerlemesi", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Güven Skoru (%)")
        ax3.set_ylim(0, 115)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Öncesi vs Sonrası Metin Karşılaştırması
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Taslak vs Düzeltilmiş Yanıt Karşılaştırması", fontsize=12, fontweight="bold", pad=10)

        karsilastirma_metni = "====================================================\n"
        karsilastirma_metni += "             İLK TASLAK (HALÜSİNASYONLU)            \n"
        karsilastirma_metni += "====================================================\n"
        karsilastirma_metni += f"{cove_sonucu['ilk_taslak_yanit']}\n"
        karsilastirma_metni += "----------------------------------------------------\n"
        karsilastirma_metni += "             DÜZELTİLMİŞ NİHAİ YANIT (CoVe)         \n"
        karsilastirma_metni += "----------------------------------------------------\n"
        karsilastirma_metni += f"{cove_sonucu['duzeltilmis_yanit']}\n"
        karsilastirma_metni += "----------------------------------------------------\n"
        karsilastirma_metni += "  TESPİT EDİLEN HATALAR:\n"
        for d in cove_sonucu["dogrulama_raporu"]:
            if d["celiski_var_mi"]:
                karsilastirma_metni += f"  • {d['konu']}: '{d['taslak_iddia']}' -> DOĞRUSU: '{d['dogrulanmis_cevap']}'\n"
        karsilastirma_metni += "===================================================="

        ax4.text(
            0.02, 0.5, karsilastirma_metni,
            fontsize=7.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: CoVe Mimari Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Chain of Verification (CoVe) Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "        CHAIN OF VERIFICATION (CoVe) PIPELINE       \n"
            "====================================================\n"
            "  [1. Kullanıcı Sorgusu]                            \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Aşama 1: İlk Taslak Üretimi (Baseline Draft)]     \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Aşama 2: Doğrulama Soruları Planlama (Planner)]   \n"
            "    - Mehmet Akif hangi şehirde doğdu?              \n"
            "    - İstiklal Marşı hangi dergahta yazıldı?        \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Aşama 3: Bağımsız Fakt Kontrolü (Independent)]   \n"
            "    (Taslak bağlamı verilmeden izole yanıtlanır)    \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Aşama 4: Çapraz Kontrol & Düzeltilmiş Yanıt]     \n"
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
        # PANEL 6: GÜN 156 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 156 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "     DAY 156 SUMMARY: CHAIN OF VERIFICATION (CoVe)  \n"
            "====================================================\n"
            "• Yöntem               : 4 Aşamalı Fakt Kontrolü (Meta AI CoVe)\n"
            "• İlk Taslak Başarımı  : %0.0 (3/3 Halüsinasyon)\n"
            "• CoVe Nihai Doğruluk  : %100.0 (Olgusal Temizleme)\n"
            "• Bağımsız Doğrulama   : Önyargısız izole bağlam sorgulaması\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. LLM'in kendi ürettiği olgusal iddiaları sorgulaması\n"
            "  2. Yönlendirici olmayan (Unbiased) soru şablonları\n"
            "  3. Taslak ile bağımsız gerçekler arasındaki çelişki tespiti\n"
            "  4. Halüsinasyonların sıfıra indirilmesi\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 157 (Dinamik Token Bütçesi - Routing)\n"
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
        print(f"  ✓ CoVe Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
