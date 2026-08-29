"""
Mistral SWA ve Rolling Buffer Cache Teşhis Panosu Görselleştirici Modülü (Day 105).
6-panelli mimari karşılaştırma, bantlı maske matrisi, sabit bellek ve alıcı alan panosu üretir.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import torch

from .sliding_window_attention import olustur_bant_maskesi


class SWAGorsellestirici:
    """Sliding Window Attention analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        gecikme_raporu: Dict[str, Dict[str, Any]],
        bellek_raporu: Dict[str, List[float]],
        alici_alan_raporu: Dict[str, Any],
        dizi_uzunluklari: List[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
        kayit_yolu: str = "ciktilar/swa_rolling_cache_paneli.png",
    ):
        """6 panelli SWA teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Mistral Sliding Window Attention (SWA) & Rolling Buffer Cache — Sabit Bellek & Alıcı Alan Analizi",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        baglam_k = [f"{int(b/1024)}k" if b >= 1024 else str(b) for b in dizi_uzunluklari]
        renk_full = "#e74a3b"
        renk_swa = "#1cc88a"

        # -------------------------------------------------------------
        # PANEL 1: KV Cache Bellek Sabitlenmesi (MB)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        keys = list(bellek_raporu.keys())
        ax1.plot(baglam_k, bellek_raporu[keys[0]], marker="o", lw=2.5, color=renk_full, label=keys[0])
        ax1.plot(baglam_k, bellek_raporu[keys[1]], marker="s", lw=3.0, color=renk_swa, label=f"{keys[1]} (SABİT!)")

        ax1.set_title("1. Bağlam Uzadıkça KV Cache (MB) — Sabit Bellek Tavanı", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Bağlam Uzunluğu (Token)")
        ax1.set_ylabel("KV Cache (MB)")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 2: Katman Sayısına Göre Etkin Alıcı Alan (Receptive Field)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        katmanlar = list(range(1, alici_alan_raporu["toplam_katman"] + 1))
        alanlar = [a / 1024.0 for a in alici_alan_raporu["katman_bazli_alanlar"]]

        ax2.plot(katmanlar, alanlar, color="#4e73df", lw=3.0, marker="^", label="Teorik Alıcı Alan ($L \times W$)")
        ax2.fill_between(katmanlar, 0, alanlar, color="#4e73df", alpha=0.15)
        ax2.set_title(f"2. Katman İstiflendikçe Alıcı Alan (W={alici_alan_raporu['pencere_boyutu']})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Transformer Katman Sayısı ($L$)")
        ax2.set_ylabel("Etkin Alıcı Alan (k Token)")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 3: Bantlı Nedensel Maske Matrisi Görseli (Heatmap)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        demo_seq = 24
        demo_w = 6
        maske_demo = olustur_bant_maskesi(demo_seq, demo_w, device=torch.device("cpu")).numpy()
        maske_binary = np.where(maske_demo == 0.0, 1.0, 0.0)

        cax3 = ax3.imshow(maske_binary, cmap="Blues", interpolation="nearest")
        ax3.set_title(f"3. SWA Bantlı Nedensel Maske Matrisi (W={demo_w}, S={demo_seq})", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Key / Value İndeksi ($j$)")
        ax3.set_ylabel("Query İndeksi ($i$)")

        # -------------------------------------------------------------
        # PANEL 4: 32k Bağlamda VRAM Tasarrufu (GB)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        full_32k_gb = bellek_raporu[keys[0]][-1] / 1024.0
        swa_32k_gb = bellek_raporu[keys[1]][-1] / 1024.0

        bars4 = ax4.bar(["Full Causal Attention", "Mistral SWA (Rolling Cache)"], [full_32k_gb, swa_32k_gb], color=[renk_full, renk_swa], width=0.5, edgecolor="black", alpha=0.85)
        ax4.set_title("4. 32,768 Token Bağlamda KV Cache VRAM (GB)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("KV Cache (GB) — Düşük Daha İyi")
        ax4.set_ylim(0, max(full_32k_gb, swa_32k_gb) * 1.35)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2, h + 0.25, f"{h:.3f} GB", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 5: Mistral SWA ve Rolling Buffer Formül Kartı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. SWA ve Rolling Buffer Matematik Kartı", fontsize=12, fontweight="bold", pad=10)

        formuller = (
            "[i] Mistral-7B Sliding Window Attention (SWA):\n"
            "--------------------------------------------------\n"
            "1. Bantlı Dikkat Kısıtı (Banded Mask):\n"
            "   Token_i sadece j in [i - W + 1, i] aralığına bakar.\n"
            "   Hesaplama: O(S^2) yerine O(S · W) doğrusal maliyet!\n\n"
            "2. Rolling Buffer Cache (Modulo İndeksleme):\n"
            "   Slot = t mod W\n"
            "   Eski token'lar dairesel olarak ezilir.\n"
            "   Bellek boyutu O(W) sabitinde kilitlenir!\n\n"
            "3. Etkin Alıcı Alan (Effective Receptive Field):\n"
            "   32 Katman x 4096 Pencere = 131,072 Token (128k)!"
        )

        ax5.text(
            0.05, 0.5, formuller,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & SWA Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & Mistral SWA Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "          MISTRAL SWA DECISION CERTIFICATE          \n"
            "====================================================\n"
            "• Soru: Eski token'ları silersek model unutmaz mı?  \n"
            "• Cevap: Hayır! Katman 1 token 0..W'yi özetler;     \n"
            "         Katman 2 bu özeti W..2W'ye taşır;         \n"
            "         32. katmana gelindiğinde model 131k token'ı\n"
            "         dolaylı olarak eksiksiz görür!             \n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Mistral-7B & Mixtral-8x7B Mimari Zaferi!\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, sertifika,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Mistral SWA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
