"""
Day 329: Neuromorphic Auditory Cochlea Filters & Event-Based Acoustic Classification
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ham ses sinyallerini, Gammatone koklear süzgeç yanıtlarını,
olay tabanlı kokleogram spike diyagramını ve SNN akustik komut teşhis panosunu barındırır.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class CochleaGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Nöromorfik Koklea ve Akustik Sınıflandırma Panosu.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cikti_dizini = os.path.join(base_dir, "ciktilar")
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    def teshis_panelini_ciz(
        self,
        raw_audio: np.ndarray,
        filtered_audio: np.ndarray,
        center_freqs: np.ndarray,
        cochleogram: np.ndarray,
        class_probs: np.ndarray,
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "koklea_isitsel_paneli.png"
    ) -> str:
        """
        6 Panelli Nöromorfik Koklea Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Neuromorphic Silicon Cochlea Filters & Event-Based Acoustic Classification Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        t_audio = np.arange(len(raw_audio)) / 16000.0  # sec

        # ------------------------------------------------------------------
        # Panel 1: Ham Ses Komutu Sinyali (Zamansal Dalga Formu)
        # ------------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(t_audio, raw_audio, color="#2c3e50", alpha=0.8, linewidth=1.2)
        ax1.set_title("1. Ham Akustik Komut Sinyali (Time-Domain Audio)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (saniye)", fontsize=8)
        ax1.set_ylabel("Genlik", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Gammatone Filtre Bankası ERB Frekans Eğrileri
        # ------------------------------------------------------------------
        ax2 = axes[0, 1]
        freq_axis = np.linspace(50, 7000, 300)
        for cf in center_freqs[:8]:
            erb = 24.7 * (4.37e-3 * cf + 1.0)
            resp = 1.0 / (1.0 + ((freq_axis - cf) / (erb * 0.5)) ** 2)
            ax2.plot(freq_axis, resp, alpha=0.7, label=f"{int(cf)} Hz")
        ax2.set_title("2. Gammatone Filtre Bankası ERB Frekans Genliği", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Frekans (Hz)", fontsize=8)
        ax2.set_ylabel("Filtre Yanıtı H(f)", fontsize=8)
        ax2.set_xscale("log")
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Çok Kanallı Süzülmüş Koklea Sinyal İzleri
        # ------------------------------------------------------------------
        ax3 = axes[0, 2]
        num_ch = min(6, filtered_audio.shape[0])
        for c in range(num_ch):
            offset = c * 1.5
            ax3.plot(t_audio[:800], filtered_audio[c, :800] + offset, label=f"Ch {c+1} ({int(center_freqs[c])}Hz)", alpha=0.85)
        ax3.set_title("3. Çok Kanallı Süzülmüş Koklear İletim İzleri", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (saniye)", fontsize=8)
        ax3.set_ylabel("Kanal Offset + Genlik", fontsize=8)
        ax3.legend(loc="upper right", fontsize=6)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Silikon Koklea Olay Tabanlı Spike Kokleogramı
        # ------------------------------------------------------------------
        ax4 = axes[1, 0]
        im4 = ax4.imshow(cochleogram, cmap="binary", aspect="auto", origin="lower")
        ax4.set_title(f"4. Silikon Koklea Spike Kokleogramı ({profiler_metrics.get('total_events', 0)} Event)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman Bin İndeksi", fontsize=8)
        ax4.set_ylabel("Koklear Kanal (ERB)", fontsize=8)

        # ------------------------------------------------------------------
        # Panel 5: SNN Akustik Sınıflandırma Olasılıkları
        # ------------------------------------------------------------------
        ax5 = axes[1, 1]
        classes = ["Evet", "Hayır", "Dur", "Geç"]
        bars = ax5.bar(classes, class_probs * 100.0, color=["#27ae60", "#e74c3c", "#f39c12", "#2980b9"], width=0.5, alpha=0.85)
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"%{yval:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. SNN Akustik Komut Tahmin Olasılıkları", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Olasılık (%)", fontsize=8)
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Nöromorfik Ses Veri Sıkıştırma ve Hazır Bulunurluk
        # ------------------------------------------------------------------
        ax6 = axes[1, 2]
        metrics_list = ["Filtre Çözünürlüğü", "Olay Sıkıştırma Kazancı", "SNN Tanıma Skoru", "Nöromorfik Koklea Sistem"]
        scores = [
            profiler_metrics.get("filter_resolution_score", 95.0),
            profiler_metrics.get("compression_score", 92.0),
            profiler_metrics.get("snn_accuracy_score", 98.0),
            profiler_metrics.get("cochlea_readiness_score", 95.0)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#3498db", alpha=0.8)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Nöromorfik Ses İşleme Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
