"""
GQA vs MHA vs MQA Teşhis Panosu Görselleştirici Modülü (Day 102).
6-panelli profesyonel mimari karşılaştırma, KV Cache bellek ayak izi ve gecikme panosu üretir.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class GQAGorsellestirici:
    """Grouped-Query Attention analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        gecikme_raporu: Dict[str, Dict[str, Any]],
        bellek_raporu: Dict[str, List[float]],
        dizi_uzunluklari: List[int] = [512, 1024, 2048, 4096, 8192],
        kayit_yolu: str = "ciktilar/gqa_mqa_kv_cache_paneli.png",
    ):
        """6 panelli GQA teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) ile KV Cache Bellek Analizi",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        modeller = list(gecikme_raporu.keys())
        kisa_etiketler = ["MHA (32 KV)", "GQA (8 KV)", "MQA (1 KV)"]
        renkler = ["#e74a3b", "#36b9cc", "#1cc88a"]

        # -------------------------------------------------------------
        # PANEL 1: KV Cache Bellek Tüketimi (MB vs Context Length)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        for (m_key, col) in zip(["MHA", "GQA", "MQA"], renkler):
            ax1.plot(dizi_uzunluklari, bellek_raporu[m_key], marker="o", lw=2.5, color=col, label=f"{m_key} Cache")

        ax1.set_title("1. Bağlam Uzunluğuna Göre KV Cache Belleği (MB)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Bağlam Uzunluğu (Token Sayısı)")
        ax1.set_ylabel("KV Cache (MB) — Düşük Daha İyi")
        ax1.set_yscale("log")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 2: P50 Çıkarım Gecikmesi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        p50_ler = [gecikme_raporu[m]["p50_ms"] for m in modeller]
        bars2 = ax2.bar(kisa_etiketler, p50_ler, color=renkler, width=0.55, edgecolor="black", alpha=0.85)
        ax2.set_title("2. P50 Çıkarım Gecikmesi (Milisaniye - ms)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Gecikme (ms) — Düşük Daha İyi")
        ax2.set_ylim(0, max(p50_ler) * 1.35)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, h + (max(p50_ler)*0.03), f"{h:.2f} ms", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: Throughput (Tokens / Saniye)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        tps_ler = [gecikme_raporu[m]["throughput_tps"] for m in modeller]
        bars3 = ax3.bar(kisa_etiketler, tps_ler, color=renkler, width=0.55, edgecolor="black", alpha=0.85)
        ax3.set_title("3. Çıkarım İşleme Hızı (Tokens / Saniye)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Throughput (Tokens/s) — Yüksek Daha İyi")
        ax3.set_ylim(0, max(tps_ler) * 1.35)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width()/2, h + (max(tps_ler)*0.03), f"{int(h):,} TPS", ha="center", fontsize=9.5, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Mimari Karşılaştırma ve Matematik Kartı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. MHA vs MQA vs GQA Mimari Formülü", fontsize=12, fontweight="bold", pad=10)

        mimari_metin = (
            "[i] Dikkat Mimarileri ve Kafa Gruplama Oranları:\n"
            "--------------------------------------------------\n"
            "1. Multi-Head Attention (MHA):\n"
            "   H_q = 32, H_kv = 32  (1:1 Eslesme)\n"
            "   Bellek: %100 | Kalite: En Yuksek | Hiz: Yavas\n\n"
            "2. Multi-Query Attention (MQA):\n"
            "   H_q = 32, H_kv = 1   (Tum Q'lar tek KV'yi paylasir)\n"
            "   Bellek: %3.1 | Kalite: Hafif Dusuk | Hiz: Cok Hizli\n\n"
            "3. Grouped-Query Attention (GQA - LLaMA-3 Standardi):\n"
            "   H_q = 32, H_kv = 8   (Her 4 Q'ya 1 KV Grubu)\n"
            "   Bellek: %25  | Kalite: MHA ile Esdeger | Hiz: %75 Hizlanma"
        )

        ax4.text(
            0.05, 0.5, mimari_metin,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: 4K Bağlamda KV Cache Bellek Tasarrufu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        mha_4k = bellek_raporu["MHA"][3]
        gqa_4k = bellek_raporu["GQA"][3]
        mqa_4k = bellek_raporu["MQA"][3]

        tasarruflar = [0.0, ((mha_4k - gqa_4k) / mha_4k) * 100.0, ((mha_4k - mqa_4k) / mha_4k) * 100.0]
        bars5 = ax5.bar(kisa_etiketler, tasarruflar, color=["#6c757d", "#36b9cc", "#1cc88a"], width=0.55, edgecolor="black", alpha=0.85)
        ax5.set_title("5. 4096 Bağlamda KV Cache VRAM Tasarrufu (%)", fontsize=12, fontweight="bold")
        ax5.set_ylabel("VRAM Tasarrufu (%) — Yüksek Daha İyi")
        ax5.set_ylim(0, 115)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width()/2, h + 2.5, f"%{h:.1f}", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & GQA Mimari Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & GQA Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        karar_kart = (
            "====================================================\n"
            "         GQA ARCHITECTURAL DECISION NOTE            \n"
            "====================================================\n"
            "• Neden MHA Terk Edildi?   : 8k+ baglamda KV Cache  \n"
            "                             VRAM'i tuketip OOM yapar.\n"
            "• Neden MQA Secilmedi?     : Kalite kaybi yasanir.  \n"
            "• Neden GQA Altin Standart?: MHA kalitesini korurken\n"
            "                             VRAM'i 4x (%75) azaltir.\n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] LLaMA-3, Mistral ve Gemma'nin Ortak     \n"
            "            Tercihi: GQA-8 (Grouped-Query Attention)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, karar_kart,
            fontsize=8.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d1ecf1", edgecolor="#17a2b8", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ GQA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
