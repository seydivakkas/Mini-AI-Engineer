"""
Vision Transformer LoRA PEFT Teşhis ve Görselleştirme Panosu
------------------------------------------------------------
6 panelli yüksek çözünürlüklü LoRA matris mimarisi, parametre verimliliği,
farklı derecelerin (rank r) karşılaştırması ve ağırlık birleştirme analiz paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LoRAGorsellestirici:
    """
    LoRA PEFT mimarisini ve ince ayar dinamiklerini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_peft_paneli(
        self,
        parametre_istatistikleri: Dict[str, Any],
        rank_ablasyonu: Dict[str, Dict[str, float]],
        gecikme_verileri: Dict[str, float],
        egitim_gecmisi: Dict[str, List[float]],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı LoRA PEFT teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 81: Vision Transformer İçin LoRA (Low-Rank Adaptation) ile Parametre-Verimli İnce Ayar (PEFT) Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: LoRA Düşük Dereceli Matris Ayrışımı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        lora_metin = (
            "          LoRA (LOW-RANK ADAPTATION) HESAPLAMA AKIŞI\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. DONDURULMUŞ TEMEL MATRİS (Frozen Base Weight):\n"
            "     • W_0 ∈ ℝ^(d_out × d_in)  ──> requires_grad = False (%98+)\n\n"
            "  2. DÜŞÜK DERECELİ ADAPTÖR MATRİSLERİ (Trainable LoRA):\n"
            "     • A ∈ ℝ^(r × d_in)   ──> Kaiming Uniform ile başlar\n"
            "     • B ∈ ℝ^(d_out × r)  ──> SIFIR (0) ile başlar!\n\n"
            "  3. İLERİ GEÇİŞ (Forward Pass):\n"
            "     • h = W_0 x + (α / r) · (B · A) x\n\n"
            "  4. DAĞITIMDA AĞIRLIK BİRLEŞTİRME (Weight Merging):\n"
            "     • W_merged = W_0 + (α / r) · (B · A)\n"
            "     • Çıkarım Zamanında: 0 ms EK HESAPLAMA GECİKMESİ!"
        )
        ax1.text(
            0.5, 0.5, lora_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f0fff4", edgecolor="#38a169", linewidth=1.8)
        )
        ax1.set_title("1. LoRA Matematiksel Matris Ayrışımı", fontsize=12, fontweight="bold", color="#276749")

        # -------------------------------------------------------------
        # PANEL 2: Parametre Dağılımı (Dondurulmuş vs Eğitilebilir)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        dondurulan = parametre_istatistikleri["dondurulan_param"]
        egitilebilir = parametre_istatistikleri["egitilebilir_param"]
        
        sizes = [dondurulan, egitilebilir]
        labels = [f"Dondurulmuş Omurga\n({dondurulan:,} param)", f"Eğitilebilir LoRA + Head\n({egitilebilir:,} param)"]
        colors = ["#a0aec0", "#48bb78"]
        explode = (0, 0.15)

        ax2.pie(
            sizes, explode=explode, labels=labels, autopct="%1.2f%%",
            startangle=140, colors=colors, textprops=dict(fontweight="bold")
        )
        ax2.set_title(
            f"2. Parametre Verimliliği (%{parametre_istatistikleri['egitilebilir_yuzde']:.2f} Eğitilebilir)",
            fontsize=12, fontweight="bold", color="#2d3748"
        )

        # -------------------------------------------------------------
        # PANEL 3: Farklı LoRA Dereceleri (Rank r) Analizi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ranks = list(rank_ablasyonu.keys())
        params = [rank_ablasyonu[r]["param_sayisi"] for r in ranks]
        accs = [rank_ablasyonu[r]["dogruluk"] for r in ranks]

        ax3_twin = ax3.twinx()
        b = ax3.bar(ranks, params, color="#90cdf4", width=0.45, label="Eğitilebilir Parametre")
        l = ax3_twin.plot(ranks, accs, "ro-", linewidth=2.2, label="Downstream Doğruluk (%)")

        ax3.set_title("3. Farklı Dereceler (Rank r=2..16) Parametre & Doğruluk", fontsize=12, fontweight="bold", color="#2b6cb0")
        ax3.set_xlabel("LoRA Derecesi (Rank r)", fontsize=10)
        ax3.set_ylabel("Parametre Sayısı", color="#2b6cb0", fontsize=10)
        ax3_twin.set_ylabel("Doğruluk (%)", color="r", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Ağırlık Birleştirme (Merging) Çıkarım Gecikmesi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        modlar = list(gecikme_verileri.keys())
        ms_sureler = list(gecikme_verileri.values())
        bar_colors = ["#f56565", "#ed8936", "#48bb78"]

        bars4 = ax4.bar(modlar, ms_sureler, color=bar_colors, width=0.45, edgecolor="#2d3748")
        ax4.set_title("4. 1000 İleri Geçiş İçin Çıkarım Süresi (ms)", fontsize=12, fontweight="bold", color="#c53030")
        ax4.set_ylabel("Gecikme (Milisaniye)", fontsize=10)
        ax4.set_ylim(0, max(ms_sureler) * 1.25)

        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f} ms", ha="center", va="bottom", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 5: LoRA İnce Ayar Eğitim Eğrisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        epoklar = list(range(1, len(egitim_gecmisi["kayip"]) + 1))
        ax5_twin = ax5.twinx()

        l1 = ax5.plot(epoklar, egitim_gecmisi["kayip"], "b-o", label="LoRA Fine-tuning Kaybı", linewidth=2)
        l2 = ax5_twin.plot(epoklar, egitim_gecmisi["dogruluk"], "g--s", label="Val Top-1 Doğruluk (%)", linewidth=2)

        ax5.set_title("5. LoRA İnce Ayar (Fine-Tuning) Yakınsama Profili", fontsize=12, fontweight="bold", color="#2c5282")
        ax5.set_xlabel("Epok", fontsize=10)
        ax5.set_ylabel("Kayıp", color="b", fontsize=10)
        ax5_twin.set_ylabel("Doğruluk (%)", color="g", fontsize=10)

        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax5.legend(lns, labs, loc="center right")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "           ViT LoRA PEFT SWOT KARAR MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Parametrelerin yalnızca ~%1-2'si eğitilerek %99+ başarı elde edilir.\n"
            "  • Ağırlık birleştirme (Weight Merging) ile çıkarımda 0 ms ek gecikme.\n"
            "  • Çoklu downstream görevler için sadece küçük adaptör dosyaları (<100KB).\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Hangi modüllerin (w_q, w_v vs ffn) seçileceği deneysel karar gerektirir.\n"
            "  • Düşük derecede (r=1,2) aşırı karmaşık görevlerde kapasite sınırı.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Kenar cihazlarda (Edge AI) ve istemcilerde hafif adaptör değişimi.\n"
            "  • QLoRA (4-bit kuantizasyon) ile bellek ayak izini %80 azaltma.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Adaptör ağırlıkları birleştirilmeden eşzamanlı çoklu çıkarımda ek yük."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Vision Transformer LoRA SWOT Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
