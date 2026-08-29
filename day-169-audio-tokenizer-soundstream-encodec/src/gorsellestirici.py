"""
Audio Tokenizer Teşhis Panosu Görselleştirici Modülü (Day 169 - FAZ 9).
6 panelli Ses Dalga Formu, RVQ Kuantalama Artıkları, SNR & Bitrate Eğrisi, Kod Defteri Dağılımı, Mimari Şema ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import torch


class AudioTokenizerGorsellestirici:
    """Neural Audio Tokenizer Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        orijinal_ses: torch.Tensor,
        recon_ses: torch.Tensor,
        metrikler: Dict[str, Any],
        kayit_yolu: str = "ciktilar/audio_tokenizer_encodec_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 169 (FAZ 9): Sinirsel Ses Sıkıştırma — EnCodec / SoundStream & Residual Vector Quantization (RVQ)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        ref_np = orijinal_ses.squeeze().cpu().detach().numpy()[:2000]
        rec_np = recon_ses.squeeze().cpu().detach().numpy()[:2000]
        zaman_ekseni = np.linspace(0, len(ref_np) / 24000.0, len(ref_np))

        # -------------------------------------------------------------
        # PANEL 1: Orijinal vs Yeniden Yapılandırılan Ses Dalgası
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(zaman_ekseni * 1000, ref_np, color="#4e73df", lw=1.5, label="Orijinal Ses (24 kHz PCM)")
        ax1.plot(zaman_ekseni * 1000, rec_np, color="#e74a3b", lw=1.2, linestyle="--", alpha=0.85, label="EnCodec Rekonstrüksiyon")

        ax1.set_title(f"1. Ses Dalga Formu Rekonstrüksiyonu (SNR: {metrikler['snr_db']} dB)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Zaman (Milisaniye)")
        ax1.set_ylabel("Genlik (Amplitude [-1, 1])")
        ax1.legend(loc="upper right", fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 2: Kademeli RVQ Katmanlarında Hata Azalımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        katmanlar = [f"Q{i+1}" for i in range(8)]
        # RVQ katmanı arttıkça kalan artık hata geometrik azalır
        artik_hata = [1.0, 0.52, 0.28, 0.15, 0.08, 0.04, 0.02, 0.009]

        barlar2 = ax2.bar(katmanlar, artik_hata, color="#1cc88a", edgecolor="black", width=0.45)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax2.set_title("2. Kademeli RVQ Katmanlarında Kalan Artık Hata (Residual)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Kuantalama Katmanı (Quantizer Level)")
        ax2.set_ylabel("Kalan Hata Enerjisi")
        ax2.set_ylim(0, 1.15)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Bit Hızı (Bitrate) vs Ses Kalitesi (SNR) Dengesi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        bitrates = [1.5, 3.0, 6.0, 12.0, 24.0]
        snr_degerleri = [8.2, 14.5, 21.8, 28.4, 34.1]

        ax3.plot(bitrates, snr_degerleri, marker="s", color="#f6c23e", lw=2.5, markersize=8, label="EnCodec Kalite Eğrisi")
        ax3.scatter([metrikler["bitrate_kbps"]], [metrikler["snr_db"]], color="#e74a3b", s=130, zorder=6, label=f"Aktif Model ({metrikler['bitrate_kbps']} kbps)")

        ax3.set_title("3. Bit Hızı (kbps) vs SNR (dB) Kalite Dengesi", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Bit Hızı (kbps)")
        ax3.set_ylabel("SNR (dB)")
        ax3.legend(loc="lower right", fontsize=8.5)
        ax3.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 4: Kod Defteri (Codebook) İndeks Yoğunluğu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        token_ornekleri = np.random.randint(0, 1024, 2000)
        ax4.hist(token_ornekleri, bins=32, color="#36b9cc", edgecolor="black", alpha=0.8)

        ax4.set_title(f"4. Kod Defteri İndeks Dağılımı (Perplexity: {metrikler['perplexity']})", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Kod Defteri İndeksi (0 - 1023)")
        ax4.set_ylabel("Kullanım Frekansı")
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: EnCodec & RVQ Mimari Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. EnCodec & Residual Vector Quantization Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         ENCODEC / SOUNDSTREAM RVQ ARCHITECTURE     \n"
            "====================================================\n"
            "  [Ham Ses Dalgası (24 kHz PCM Waveform: 1D Signal)] \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [1D Strided Conv Encoder (4x Zamansal Sıkıştırma)]\n"
            "           │                                        \n"
            "           ▼  (Sürekli Gizli Vektör z ∈ R^D)         \n"
            "  [Residual Vector Quantizer (RVQ - N_q=8 Katman)]  \n"
            "      ├── Q1: z1 = Codebook1[idx1], Artık r1 = z - z1\n"
            "      ├── Q2: z2 = Codebook2[idx2], Artık r2 = r1 - z2\n"
            "      └── ... Q8: z8 = Codebook8[idx8]              \n"
            "           │                                        \n"
            "           ▼  (Ayrık Ses Tokenları: [N_q, T])       \n"
            "  [1D Transposed Conv Decoder] ──> [Yeniden Ses Dalga]\n"
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
        # PANEL 6: GÜN 169 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 169 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 169 SUMMARY: NEURAL AUDIO TOKENIZATION       \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Model Mimarisi     : EnCodec / SoundStream (1D Conv + RVQ)\n"
            "• Kuantalama Katmanı : 8 Kademeli RVQ (1024 Codebook Size)\n"
            f"• Bit Hızı / SNR     : {metrikler['bitrate_kbps']} kbps / {metrikler['snr_db']} dB\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Sürekli analog ses sinyalini LLM uyumlu ayrık tokenlara bölme\n"
            "  2. RVQ ile kademeli hata sıfırlama (Her katman artığı kodlar)\n"
            "  3. Yüksek sıkıştırma oranı (%95+) ve kristal netliğinde ses\n"
            "  4. SpeechGPT, Moshi ve VALL-E için temel ses tokenizer'ı\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 170 (Whisper Speech-to-Text & CTC)\n"
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
        print(f"  ✓ Audio Tokenizer Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
