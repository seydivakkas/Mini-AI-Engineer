"""
DeepSeek MLA Teşhis Panosu Görselleştirici Modülü (Day 103).
6-panelli mimari karşılaştırma, sıkıştırılmış KV latent analizi ve bellek tasarrufu panosu üretir.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class MLAGorsellestirici:
    """DeepSeek Multi-Head Latent Attention (MLA) analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        gecikme_raporu: Dict[str, Dict[str, Any]],
        bellek_raporu: Dict[str, List[float]],
        dizi_uzunluklari: List[int] = [512, 1024, 2048, 4096, 8192, 16384, 32768],
        kayit_yolu: str = "ciktilar/deepseek_mla_teshis_paneli.png",
    ):
        """6 panelli MLA teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DeepSeek Multi-Head Latent Attention (MLA) — Sıkıştırılmış KV Latent & Bellek Analizi",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        renkler = {"MHA (16 KV Kafa)": "#e74a3b", "GQA (4 KV Kafa)": "#f6c23e", "DeepSeek MLA": "#4e73df"}

        # -------------------------------------------------------------
        # PANEL 1: KV Cache Bellek Tüketimi (Log Scale)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        for m_key in bellek_raporu.keys():
            col = renkler.get(m_key, "#1cc88a")
            ax1.plot(dizi_uzunluklari, bellek_raporu[m_key], marker="s", lw=2.5, color=col, label=m_key)

        ax1.set_title("1. Bağlam Uzunluğuna Göre KV Cache (MB)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Bağlam Uzunluğu (Token)")
        ax1.set_ylabel("Bellek Ayak İzi (MB) — Düşük Daha İyi")
        ax1.set_yscale("log")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper left")

        # -------------------------------------------------------------
        # PANEL 2: Token Başına Önbellek Eleman Sayısı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        modeller = ["MHA (16 Kafa)", "GQA (4 Kafa)", "DeepSeek MLA"]
        # MHA: 2 * 16 * 32 = 1024, GQA: 2 * 4 * 32 = 256, MLA: 128 (d_c) + 32 (d_R) = 160
        eleman_sayilari = [1024, 256, 160]
        bar_renkleri = ["#e74a3b", "#f6c23e", "#4e73df"]
        bars2 = ax2.bar(modeller, eleman_sayilari, color=bar_renkleri, width=0.55, edgecolor="black", alpha=0.85)

        ax2.set_title("2. Token Başına Önbellek Boyutu (Eleman Sayısı)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Eleman / Token — Düşük Daha İyi")
        ax2.set_ylim(0, max(eleman_sayilari) * 1.35)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2, h + 25, f"{h} Eleman", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: MLA Mimari ve Matematik Şeması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.axis("off")
        ax3.set_title("3. DeepSeek MLA Matematiksel Formülü", fontsize=12, fontweight="bold", pad=10)

        mla_formulu = (
            "[i] DeepSeek-V2 / DeepSeek-V3 MLA Formülleri:\n"
            "--------------------------------------------------\n"
            "1. Düşük Dereceli KV Ortak Sıkıştırması:\n"
            "   c_t^{KV} = h_t · W_{DKV}  (d -> d_c)\n"
            "   Bellekte SADECE c_t^{KV} saklanır! (Up-proj yok)\n\n"
            "2. Ayrık RoPE (Decoupled Position):\n"
            "   k_t^R = RoPE(h_t · W_{KR})  (d -> d_R)\n\n"
            "3. Matris Soğurma (Matrix Absorption):\n"
            "   Q' = Q · W_{UK}^T  (Çıkarımda KV açılımı sıfır!)\n"
            "   Skor = (Q' · (c^{KV})^T + Q_R · (K_R)^T) / sqrt(d_h)"
        )

        ax3.text(
            0.05, 0.5, mla_formulu,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#4e73df", lw=1.8),
        )

        # -------------------------------------------------------------
        # PANEL 4: 32k Bağlamda Toplam KV Cache Belleği (GB)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        mha_32k_gb = bellek_raporu["MHA (16 KV Kafa)"][-1] / 1024.0
        gqa_32k_gb = bellek_raporu["GQA (4 KV Kafa)"][-1] / 1024.0
        mla_32k_gb = bellek_raporu["DeepSeek MLA"][-1] / 1024.0

        gb_ler = [mha_32k_gb, gqa_32k_gb, mla_32k_gb]
        bars4 = ax4.bar(modeller, gb_ler, color=bar_renkleri, width=0.55, edgecolor="black", alpha=0.85)
        ax4.set_title("4. 32,768 Token Bağlamda VRAM Tüketimi (GB)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("KV Cache (GB) — Düşük Daha İyi")
        ax4.set_ylim(0, max(gb_ler) * 1.35)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2, h + 0.3, f"{h:.2f} GB", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 5: MHA ve GQA'ya Göre VRAM Tasarruf Oranları
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        tasarruf_mha = ((mha_32k_gb - mla_32k_gb) / mha_32k_gb) * 100.0
        tasarruf_gqa = ((gqa_32k_gb - mla_32k_gb) / gqa_32k_gb) * 100.0

        tasarruflar = [tasarruf_mha, tasarruf_gqa]
        karsilastirmalar = ["MLA vs MHA", "MLA vs GQA-4"]
        bars5 = ax5.bar(karsilastirmalar, tasarruflar, color=["#1cc88a", "#36b9cc"], width=0.5, edgecolor="black", alpha=0.85)
        ax5.set_title("5. DeepSeek MLA'nın VRAM Tasarruf Oranı (%)", fontsize=12, fontweight="bold")
        ax5.set_ylabel("Tasarruf Oranı (%) — Yüksek Daha İyi")
        ax5.set_ylim(0, 115)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width()/2, h + 2.5, f"%{h:.1f}", ha="center", fontsize=10, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: Stajyer Notu & DeepSeek MLA Karar Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. Stajyer Notu & MLA Karar Sertifikası", fontsize=12, fontweight="bold", pad=10)

        sertifika_metin = (
            "====================================================\n"
            "       DEEPSEEK MLA ARCHITECTURE CERTIFICATE        \n"
            "====================================================\n"
            "• GQA Neden Yetersiz Kaldı? : 128 kafa modelde GQA\n"
            "                             bile 128k bağlamda devasa\n"
            "                             bellek harcar.\n"
            "• MLA'nın Çözümü            : KV tensörlerini düşük\n"
            "                             dereceli latent uzayına\n"
            "                             (d_c) sıkıştırmak.\n"
            "• Matris Soğurma Avantajı   : Çıkarımda KV açılımı\n"
            "                             yapılmadan doğrudan iç\n"
            "                             çarpım hesaplanır.\n"
            "----------------------------------------------------\n"
            "[SONUÇ] DeepSeek-V3 & R1 Standardı Onaylandı!\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, sertifika_metin,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#e8f4f8", edgecolor="#17a2b8", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ DeepSeek MLA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
