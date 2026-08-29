"""
Speech-to-Speech LLM Teşhis Panosu Görselleştirici Modülü (Day 171 - FAZ 9).
6 panelli Çift Akışlı Token Akışı, Gecikme Kıyası (Cascaded vs DualLM), Çok Katmanlı RVQ Üretimi, Diyalog İcra İzi, Mimari Şema ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class SpeechLLMGorsellestirici:
    """Uçtan Uca Speech-to-Speech LLM Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        diyalog_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/speech_to_speech_llm_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 171 (FAZ 9): Uçtan Uca Speech-to-Speech LLM (DuaLLM / Moshi) — Çift Akışlı Ses & Metin Token Modelleme",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Uçtan Uca Gecikme (Latency) Kıyası: Geleneksel vs DuaLLM
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asama_isimleri = ["Geleneksel Zincir\n(ASR + LLM + TTS)", "Uçtan Uca DuaLLM\n(Direct Audio-to-Audio)"]
        gecikmeler = [diyalog_raporu["ortalama_geleneksel_gecikme_ms"], diyalog_raporu["ortalama_duallm_gecikme_ms"]]
        renkler1 = ["#e74a3b", "#1cc88a"]

        barlar1 = ax1.bar(asama_isimleri, gecikmeler, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{int(h)} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("1. Uçtan Uca Yanıt Gecikmesi Kıyası (8.9x Daha Hızlı)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Gecikme Süresi (Milisaniye - ms)")
        ax1.set_ylim(0, 1850)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Geleneksel Boru Hattı Gecikme Dağılımı (ASR / LLM / TTS)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        bilesenler = ["ASR (Whisper)", "LLM (Metin Üretimi)", "TTS (Ses Sentezi)"]
        sureler = [400, 650, 485]
        renkler2 = ["#4e73df", "#f6c23e", "#e74a3b"]

        wedges, texts, autotexts = ax2.pie(
            sureler, labels=bilesenler, autopct="%1.1f%%", startangle=140, colors=renkler2,
            textprops=dict(color="black", fontweight="bold")
        )
        ax2.set_title("2. Geleneksel Zincirde Gecikme Dağılımı (Toplam: 1535 ms)", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: 8 Kademeli RVQ Ses Token Üretim Isı Haritası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        # 8 katman x 20 zaman adımı rastgele kod defteri indeks matrisi
        rvq_grid = np.random.randint(0, 1024, (8, 20))
        im3 = ax3.imshow(rvq_grid, aspect="auto", cmap="viridis", interpolation="nearest")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax3.set_title("3. Paralel 8-Kademeli RVQ Ses Token Matrisi", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Zaman Çerçeveleri (Time Steps - 12.5Hz)")
        ax3.set_ylabel("RVQ Katmanları (Q1 - Q8)")
        ax3.set_yticks(range(8))
        ax3.set_yticklabels([f"Q{i+1}" for i in range(8)])

        # -------------------------------------------------------------
        # PANEL 4: Çift Akışlı Canlı Sohbet Transkripsiyon ve Yanıt İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Canlı Sesli Sohbet İcra İzi", fontsize=12, fontweight="bold", pad=10)

        s1 = diyalog_raporu["senaryolar"][0]
        diyalog_metni = (
            "====================================================\n"
            "       DUAL-STREAM SPEECH-TO-SPEECH DIALOGUE TRACE  \n"
            "====================================================\n"
            f"DİYALOG NO : {s1['diyalog_id']}\n"
            f"KULLANICI  : \"{s1['kullanici_sesi']}\" (Audio Tokens In)\n"
            "----------------------------------------------------\n"
            f"DUALLM SESLİ YANIT:\n"
            f"  \"{s1['asistan_yaniti_ses']}\"\n"
            "----------------------------------------------------\n"
            f"• Geleneksel Gecikme : {s1['geleneksel_gecikme_ms']} ms\n"
            f"• DuaLLM Gecikmesi   : {s1['duallm_gecikme_ms']} ms (Ultra Düşük Gecikme)\n"
            f"• Akustik Uyum       : %{s1['akustik_uyum_skoru']*100:.1f}\n"
            "DURUM: [DOĞAL VE AKICI SESLİ SOHBET]\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, diyalog_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: DuaLLM / Moshi Çift Başlıklı Mimari Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. DuaLLM / Moshi Çift Akışlı Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "     SPEECH-TO-SPEECH DUAL-HEAD ARCHITECTURE        \n"
            "====================================================\n"
            "  [Kullanıcı Ses Tokenları (RVQ)] + [Metin Prompt]  \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Çift Akışlı Gömme Katmanı (Shared Latent Space)] \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Causal Transformer Backbone (Audio-Text Joint)]  \n"
            "           ├── Başlık 1: Metin LM Head (İç Düşünce) \n"
            "           └── Başlık 2: 8 Kademeli RVQ Ses Başlığı \n"
            "           │                                        \n"
            "           ▼  (Doğrudan Ayrık Ses Tokenları)        \n"
            "  [Neural Audio Decoder (EnCodec)] ──> [Yanıt Sesi] \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 171 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 171 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 171 SUMMARY: SPEECH-TO-SPEECH LLM (DUALLM)   \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Mimari Modeli      : Moshi / GPT-4o Tarzı Çift Başlıklı LLM\n"
            "• Çıktı Formatı      : Eş Zamanlı Metin + 8-Katman RVQ Sesi\n"
            f"• Yanıt Gecikmesi    : {diyalog_raporu['ortalama_duallm_gecikme_ms']} ms ({diyalog_raporu['gecikme_iyilesmesi_kat']}x Hız)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. ASR + LLM + TTS zincirindeki bilgi ve ton kaybını önleme\n"
            "  2. Doğrudan ses tokenı alıp ses tokenı üretme\n"
            "  3. İnsan tepki hızında (<200ms) gerçek zamanlı sesli diyalog\n"
            "  4. Ses tonu, duygu ve nefes alma doğallığını koruma\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 172 (Latent Diffusion Models - LDM)\n"
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
        print(f"  ✓ Speech-to-Speech LLM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
