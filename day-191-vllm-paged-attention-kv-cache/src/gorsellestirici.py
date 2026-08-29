"""
vLLM PagedAttention 6 Panelli Görselleştirici Modülü (Day 191 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class PagedAttentionGorsellestirici:
    """PagedAttention 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        istek_analizi: Dict[str, Any],
        tarama_raporu: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/paged_attention_paneli.png",
    ):
        """6 Panelli PagedAttention Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 191: VLLM MİMARİSİ — PAGEDATTENTION İLE SIFIR BELLEK PARÇALANMASI VE DİNAMİK KV CACHE",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Sanal Bellek Sayfalama ve Blok Tablosu
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asama_adlari = ["1. Prompt Girişi", "2. Mantıksal Sayfalama", "3. Blok Tablosu Eşleme", "4. Dağınık Fiziksel Blok", "5. CoW Paylaşımı"]
        bellek_verimi = [0.8, 1.4, 1.9, 2.3, 2.7]
        bar_renkler1 = ["#3b82f6", "#6366f1", "#8b5cf6", "#10b981", "#f59e0b"]

        ax1.barh(asama_adlari, bellek_verimi, color=bar_renkler1, height=0.5, edgecolor="#ffffff")
        ax1.set_xlabel("Sanal Bellek Yürütme Katmanı", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. PagedAttention Sanal Sayfalama Mimarisi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Bellek İsrafı Kıyası (%83 İsraf -> %2.4 İsraf)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        yaklasimlar = ["Geleneksel Statik Tahsis\n(Maksimum Dizi Rezervi)", "vLLM PagedAttention\n(Dinamik Blok Tahsisi)"]
        israflar = [istek_analizi["statik_israf_yuzde"], istek_analizi["paged_israf_yuzde"]]

        bars2 = ax2.bar(yaklasimlar, israflar, color=["#ef4444", "#10b981"], width=0.45)
        ax2.set_ylabel("VRAM İsraf Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. KV Cache Bellek İsrafı (%80+ Kazanç)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Eşzamanlı İstek Sayısı vs VRAM Tüketimi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        istekler = [r["istek_sayisi"] for r in tarama_raporu]
        statik_gb = [r["statik_vram_gb"] for r in tarama_raporu]
        paged_gb = [r["paged_vram_gb"] for r in tarama_raporu]

        ax3.plot(istekler, statik_gb, marker="o", color="#ef4444", linewidth=2.5, label="Geleneksel Statik KV Cache")
        ax3.plot(istekler, paged_gb, marker="s", color="#10b981", linewidth=2.5, label="vLLM PagedAttention KV Cache")
        ax3.set_xlabel("Eşzamanlı İstek Sayısı (Concurrent Requests)", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Toplam KV Cache VRAM (GB)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Eşzamanlı İstek Ölçeklenebilirliği", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: Copy-on-Write (CoW) Paylaşımlı Prompt Belleği
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        orneklemeler = ["1 Yanıt (Tekil)", "2 Paralel Yanıt", "4 Paralel Yanıt", "8 Paralel Yanıt"]
        cow_tasarruf = [0.0, 48.0, 72.0, 84.0]

        bars4 = ax4.bar(orneklemeler, cow_tasarruf, color="#0284c7", width=0.45)
        ax4.set_ylabel("Prompt Bellek Tasarrufu (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Copy-on-Write (CoW) Paylaşım Kazancı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.4)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{int(h)}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 5: Blok Boyutuna Göre Dahili Fragmentasyon
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        blok_boyutlari = [8, 16, 32, 64]
        fragmentasyon_oranlari = [1.6, 3.2, 6.4, 12.8]

        ax5.plot(blok_boyutlari, fragmentasyon_oranlari, marker="o", color="#f59e0b", linewidth=2.5)
        ax5.set_xlabel("Blok Boyutu (Token / Blok)", fontsize=10, color="#cbd5e1")
        ax5.set_ylabel("Dahili Fragmentasyon (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Blok Boyutu vs Dahili Parçalanma (Optimum=16)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(True, linestyle=":", alpha=0.3)

        for bl, fr in zip(blok_boyutlari, fragmentasyon_oranlari):
            ax5.text(bl, fr + 0.3, f"%{fr:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 191 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 191: VLLM PAGEDATTENTION KARNE\n"
            "----------------------------------------------------\n"
            "• Mimari              : Sanal Bellek Paging (OS Paging)\n"
            "• Blok Boyutu         : 16 Token / Fiziksel Blok\n"
            "• Bellek İsrafı       : %83.0 -> %2.4 (%80+ VRAM Kazancı)\n"
            "• Dış Fragmentasyon   : %0 (Tüm bloklar eşit boyutta)\n"
            "• Verim Artışı        : 2.5x - 4.0x Daha Fazla İstek/Saniye\n"
            "• Copy-on-Write (CoW) : Paralel Örneklemede %80+ Prompt Paylaşımı\n"
            "• Blok Tablosu        : Mantıksal Token -> Fiziksel Blok ID\n"
            "----------------------------------------------------\n"
            "SONUÇ: Modern LLM sunucularında GPU VRAM darboğazını\n"
            "yıkarak yüksek eşzamanlı çıkarımı mümkün kılan standart!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=10,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
