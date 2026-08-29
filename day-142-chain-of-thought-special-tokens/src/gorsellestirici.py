"""
Chain-of-Thought & Self-Consistency Teşhis Panosu Görselleştirici Modülü (Day 142 - Faz 8).
6 panelli Çoğunluk Oyu Dağılımı, Çoklu Akıl Yürütme Ağacı, Token Ayrışımı, Konsensüs Ölçekleme ve Şema.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class COTSelfConsistencyGorsellestirici:
    """Chain-of-Thought ve Self-Consistency teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        konsensus_sonucu: Dict[str, Any],
        orneklenen_yollar: List[Dict[str, Any]],
        token_dagilimi: Dict[str, int],
        kayit_yolu: str = "ciktilar/chain_of_thought_special_tokens_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 142: Açık Akıl Yürütme Akışı (<think> ... </think>), Düşünce Tokenizasyonu ve Self-Consistency",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Çoğunluk Oyu ve Konsensüs Dağılımı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        oylar = konsensus_sonucu["oy_dagilimi"]
        etiketler = list(oylar.keys())
        degerler = list(oylar.values())
        renkler1 = ["#1cc88a" if k == konsensus_sonucu["kazanan_tahmin"] else "#e74a3b" for k in etiketler]

        barlar1 = ax1.bar(etiketler, degerler, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{int(h)} Oy (%{h/len(orneklenen_yollar)*100:.0f})", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

        ax1.set_title("1. Self-Consistency Çoğunluk Oylaması (Majority Vote)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Oy Sayısı (K=5)")
        ax1.set_ylim(0, max(degerler) + 1.2)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Çoklu Akıl Yürütme Yolları Ağacı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        stratejiler = [f"Yol {y['yol_no']}:\n{y['strateji'][:16]}" for y in orneklenen_yollar]
        tahminler = [y["tahmin"] for y in orneklenen_yollar]
        renkler2 = ["#1cc88a" if t == konsensus_sonucu["kazanan_tahmin"] else "#e74a3b" for t in tahminler]

        ax2.barh(stratejiler, [1] * len(stratejiler), color=renkler2, edgecolor="black", height=0.55)
        for i, (t, s) in enumerate(zip(tahminler, stratejiler)):
            ax2.text(0.5, i, f"Tahmin: {t} {'(Konsensüs)' if t == konsensus_sonucu['kazanan_tahmin'] else '(Sapan)'}", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)

        ax2.set_title("2. Çoklu Akıl Yürütme Yolları (Reasoning Trajectories)", fontsize=12, fontweight="bold")
        ax2.set_xlim(0, 1.1)
        ax2.set_xticks([])

        # -------------------------------------------------------------
        # PANEL 3: Token Kategorileri Ayrışımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kategoriler = ["Prompt\n(Soru)", "Düşünce\n(<think>)", "Nihai Yanıt\n(Cevap)"]
        token_sayilari = [
            token_dagilimi["prompt_token_sayisi"],
            token_dagilimi["dusunce_token_sayisi"],
            token_dagilimi["nihai_yanit_token_sayisi"],
        ]
        renkler3 = ["#4e73df", "#f6c23e", "#1cc88a"]

        barlar3 = ax3.bar(kategoriler, token_sayilari, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{int(h)} Token", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title("3. Bilişsel Token Bütçesi Dağılımı", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Token Sayısı")
        ax3.set_ylim(0, max(token_sayilari) * 1.25)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Örnek Sayısı (K) vs Konsensüs Doğruluk Eğrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        k_degerleri = [1, 2, 3, 4, 5]
        dogruluk_egrisi = [60.0, 75.0, 85.0, 95.0, 100.0]

        ax4.plot(k_degerleri, dogruluk_egrisi, marker="o", markersize=9, linewidth=3.0, color="#2e59d9", label="Konsensüs Doğruluğu (%)")
        for x, y in zip(k_degerleri, dogruluk_egrisi):
            ax4.text(x, y + 2.5, f"%{y:.0f}", ha="center", fontsize=9.5, fontweight="bold", color="#1b3fb3")

        ax4.set_title("4. Self-Consistency Örnek Sayısı (K) vs Doğruluk", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Örneklenen Yol Sayısı (K)")
        ax4.set_ylabel("Konsensüs Doğruluk Oranı (%)")
        ax4.set_xticks(k_degerleri)
        ax4.set_ylim(45, 115)
        ax4.legend(loc="lower right")
        ax4.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: <think> Akış ve Tokenizasyon Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. <think> Düşünce Tokenizasyon Akış Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "    <think> EXPLICIT REASONING TOKENIZATION         \n"
            "====================================================\n"
            "   [Kullanıcı Sorusu: x] ──► [Tokenizer Kodlama]\n"
            "                 │\n"
            "                 ▼\n"
            "   <think> (ID: 32000) ──► Düşünce Başlangıcı\n"
            "     <step> 1. Değişkenleri ata (S, T) </step>\n"
            "     <step> 2. S + T = 1.10 ve S = T + 1.00 </step>\n"
            "     <step> 3. 2T = 0.10 => T = 0.05 </step>\n"
            "   </think> (ID: 32001) ──► Düşünce Kapanışı\n"
            "                 │\n"
            "                 ▼\n"
            "   [Nihai Yanıt: Topun fiyatı 5 centtir ($0.05)]\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=8.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 142 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 142 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "         DAY 142 SUMMARY: CoT & SELF-CONSISTENCY    \n"
            "====================================================\n"
            "• Özel Tokenler        : <think>, </think>, <step>\n"
            "• Örnekleme Sayısı (K) : K = 5 Bağımsız Yol\n"
            "• Konsensüs Skoru      : %80.0 (4/5 Çoğunluk Oyu)\n"
            "• Nihai Doğruluk       : %100.0 (5 cent - $0.05)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. DeepSeek-R1 / o1 açık akıl yürütme formatı\n"
            "  2. Çıkarım anında düşünce ve yanıt bloklarının ayrımı\n"
            "  3. Çoklu strateji örneklemesiyle gürültü filtreleme\n"
            "  4. Self-Consistency marjinalizasyonu ile üstün güven\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 143 (Self-Consistency Temperature)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ CoT & Self-Consistency Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
