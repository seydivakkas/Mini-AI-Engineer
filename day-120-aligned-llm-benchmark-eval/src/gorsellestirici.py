"""
Faz 6 Capstone Teşhis ve Liderlik Panosu Görselleştirici Modülü (Day 120).
6 panelli Chatbot Arena Elo Liderlik Tablosu, MT-Bench 8 Kategori Radar Analizi, Kazanma Oranları ve Faz 6 Mezuniyet Sertifikası.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class ArenaGorsellestirici:
    """Faz 6 modellerinin değerlendirme sonuçları için 6 panelli şampiyona panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/faz6_capstone_benchmark_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "FAZ 6 BÜYÜK FİNALİ: Aligned LLM Benchmark & Chatbot Arena Şampiyonası",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        liderlik = rapor["liderlik_tablosu"]
        model_isimleri = [item["model_adi"].replace(" Model", "").replace(" (Pretrained)", "").replace(" (Packed SFT)", "") for item in liderlik]
        elo_puanlari = [item["elo"] for item in liderlik]
        kazanma_oranlari = [item["kazanma_orani"] for item in liderlik]

        # -------------------------------------------------------------
        # PANEL 1: Chatbot Arena Elo Liderlik Sıralaması
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        y_pos = np.arange(len(model_isimleri))
        renkler1 = ["#d4af37", "#c0c0c0", "#cd7f32"] + ["#4e73df"] * (len(model_isimleri) - 3)

        barlar1 = ax1.barh(y_pos, elo_puanlari, color=renkler1, edgecolor="black")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_isimleri, fontsize=9, fontweight="bold")
        ax1.invert_yaxis()  # En yüksek Elo en üstte

        for bar in barlar1:
            w = bar.get_width()
            ax1.text(w + 10, bar.get_y() + bar.get_height() / 2, f"{w:.1f} Elo", va="center", fontweight="bold", fontsize=9)

        ax1.set_title("1. Chatbot Arena Dinamik Elo Liderlik Tablosu", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Elo Derecesi (Başlangıç: 1000)")
        ax1.set_xlim(800, max(elo_puanlari) + 120)
        ax1.grid(axis="x", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: MT-Bench 8 Kategori Skorları (Seçkin Modeller)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kats = [k.split()[0] for k in rapor["kategoriler"]]
        grpo_skorlar = list(rapor["mt_bench_kategori_skorlari"]["GRPO Model (DeepSeek-R1 Reasoning)"].values())
        dpo_skorlar = list(rapor["mt_bench_kategori_skorlari"]["DPO Model (Direct Preference)"].values())
        base_skorlar = list(rapor["mt_bench_kategori_skorlari"]["Base Model (Pretrained)"].values())

        x_kats = np.arange(len(kats))
        width = 0.25

        ax2.bar(x_kats - width, base_skorlar, width, label="Base Model", color="#e74a3b", edgecolor="black")
        ax2.bar(x_kats, dpo_skorlar, width, label="DPO Aligned", color="#4e73df", edgecolor="black")
        ax2.bar(x_kats + width, grpo_skorlar, width, label="GRPO (DeepSeek-R1)", color="#1cc88a", edgecolor="black")

        ax2.set_title("2. MT-Bench Kategori Bazlı Başarım (1-10 Puan)", fontsize=12, fontweight="bold")
        ax2.set_xticks(x_kats)
        ax2.set_xticklabels(kats, rotation=25, fontsize=8)
        ax2.set_ylabel("MT-Bench Puanı")
        ax2.set_ylim(0, 11)
        ax2.legend(loc="lower right")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Kazanma Oranları (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        barlar3 = ax3.bar(model_isimleri, kazanma_oranlari, color="#36b9cc", edgecolor="black")
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=8)

        ax3.set_title("3. Turnuva Karşılaşmaları Kazanma Oranı (%)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Kazanma Oranı (%)")
        ax3.set_ylim(0, 115)
        ax3.tick_params(axis="x", rotation=30)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Pozisyon Yanlılığı ve Swap Testi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        yanlilik_orani = rapor["pozisyon_yanliligi_tespit_orani"]
        guvenli_orani = 100.0 - yanlilik_orani

        ax4.pie(
            [guvenli_orani, yanlilik_orani],
            labels=[f"Objektif Karar\n(%{guvenli_orani:.1f})", f"Pozisyon Yanlılığı\n(Swap Düzeltildi: %{yanlilik_orani:.1f})"],
            colors=["#20c997", "#f6c23e"],
            autopct="%1.1f%%",
            startangle=140,
            explode=(0.05, 0),
            textprops={"fontweight": "bold", "fontsize": 10},
        )
        ax4.set_title("4. LLM Hakemliği Pozisyon Yanlılığı Denetimi", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 5: Faz 6 Hizalama ve Akıl Yürütme Evrim Ağacı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Faz 6 Hizalama & Akıl Yürütme Evrim Ağacı", fontsize=12, fontweight="bold", pad=10)

        evrim_metni = (
            "====================================================\n"
            "         FAZ 6 EVOLUTIONARY ALIGNMENT TREE          \n"
            "====================================================\n"
            "1. TEMEL MİMARİLER (Gün 102 - 105):\n"
            "   • GQA / MLA / RoPE & YaRN Uzun Bağlam Uzantısı\n\n"
            "2. SÜPERVİZELİ EĞİTİM & VERİMLİLİK (Gün 106 - 107):\n"
            "   • Packed SFT (Sıfır Padding) & QLoRA 4-bit NF4\n\n"
            "3. KLASİK & HAFİF TERCİH HİZALAMASI (Gün 108 - 113):\n"
            "   • Bradley-Terry Ödül Modeli -> PPO -> DPO -> KTO\n"
            "   • ORPO (Monolithic) -> SimPO (Target Margin)\n\n"
            "4. AKIL YÜRÜTME, FÜZYON & GÜVENLİK (Gün 114 - 120):\n"
            "   • GRPO (DeepSeek-R1) -> Model Merging (SLERP/TIES)\n"
            "   • Evol-Instruct -> Guardrails -> Filigran -> Distill\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, evrim_metni,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Faz 6 Mezuniyet ve SOTA Sertifikası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. FAZ 6 BÜYÜK FİNALİ MEZUNİYET SERTİFİKASI", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "    PHASE 6 CAPSTONE GRADUATION & SOTA CERTIFICATE  \n"
            "====================================================\n"
            f"• Şampiyon Model       : {rapor['sampiyon_model']}\n"
            f"• Şampiyon Elo Puanı   : {rapor['sampiyon_elo']:.1f} Elo (En Yüksek)\n"
            f"• Toplam Arena Maçı    : {rapor['toplam_mac_sayisi']} Karşılaşma\n"
            "• Faz 6 Kapsamı        : 19 Günlük İleri LLM, RLHF,  \n"
            "                         DPO, GRPO, Guardrails & KD \n"
            "• Başarı Oranı         : %100 (19/19 Gün Eksiksiz)  \n"
            "----------------------------------------------------\n"
            "[TEBRİKLER] FAZ 6 BAŞARIYLA TAMAMLANDI!\n"
            "Sıradaki Aşama: FAZ 7 - Otonom AI Ajanları & GraphRAG\n"
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
        print(f"  ✓ Faz 6 Capstone Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
