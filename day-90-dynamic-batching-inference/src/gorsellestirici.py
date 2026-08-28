"""
Dinamik Batching Teşhis ve Performans Panosu
--------------------------------------------
6 panelli yüksek çözünürlüklü Dinamik Batching Mimarisi, Throughput Kıyası,
Gecikme Dağılımı (P50/P90/P99), GPU Alt-Doğrusal Ölçeklenme ve SWOT Karar Matrisi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DinamikBatchGorsellestirici:
    """
    Dinamik batching ve ardışık çıkarım kıyaslamalarını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_batching_paneli(
        self,
        ardisik_sonuc: Dict[str, Any],
        dinamik_sonuc: Dict[str, Any],
        batch_olcekleme_verisi: Dict[str, List[float]],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Dinamik Batching Teşhis Panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 90: GPU Verimliliği İçin Kuyruk Tabanlı Dinamik Batching Çıkarım Motoru Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Dinamik Batching Mimarisi ve Tetikleme Kuralı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        mimarisi_metin = (
            "         DİNAMİK BATCHING ÇIKARIM MİMARİSİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. ASENKRON İSTEMCİ İSTEKLERİ (Client Requests):\n"
            "     • Web API'den tekil B=1 istekler FIFO kuyruğuna düşer.\n"
            "     • Her istek için anında bir `Future` nesnesi döndürülür.\n\n"
            "  2. ÇİFT EŞİKLİ TETİKLEME KURALI (Dual Threshold Trigger):\n"
            "     • Boyut Eşiği : Kuyruk boyutu >= max_batch_size (32)\n"
            "     • Zaman Eşiği : İlk eleman bekleme >= max_delay (8ms)\n"
            "     ──> İkisinden biri sağlandığı an BATCH GPU'ya fırlatılır!\n\n"
            "  3. GPU SATURATION & SLICING (Tensör Çekirdeği Doygunluğu):\n"
            "     • Tek seferde [B, 3, 32, 32] tensörü işlenir.\n"
            "     • Çıktılar tek tek dilimlenip istemcilere dağıtılır."
        )
        ax1.text(
            0.5, 0.5, mimarisi_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Kuyruk Tabanlı Dinamik Batching Mimarisi", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: İşlem Hacmi (Throughput: req/s) Kıyası
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        modlar = ["Tekil Ardışık (B=1)", "Dinamik Batching"]
        throughputs = [ardisik_sonuc["throughput_req_s"], dinamik_sonuc["throughput_req_s"]]
        renkler = ["#e53e3e", "#38a169"]

        bars = ax2.bar(modlar, throughputs, color=renkler, width=0.55, edgecolor="black")
        for bar, val in zip(bars, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, val + max(throughputs)*0.02, f"{val:.1f} req/s", ha="center", fontsize=10, fontweight="bold")

        artis_kat = dinamik_sonuc["throughput_req_s"] / max(1e-5, ardisik_sonuc["throughput_req_s"])
        ax2.set_title(f"2. İşlem Hacmi (Throughput) — {artis_kat:.1f}x Hızlanma!", fontsize=12, fontweight="bold", color="#22543d")
        ax2.set_ylabel("İstek / Saniye (Throughput)", fontsize=10)
        ax2.set_ylim(0, max(throughputs) * 1.25)

        # -------------------------------------------------------------
        # PANEL 3: Gecikme Dağılımı (P50, P90, P99 Latency)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metrikler = ["Ortalama", "P50 (Medyan)", "P90", "P99"]
        ardisik_lat = [
            ardisik_sonuc["ortalama_gecikme_ms"],
            ardisik_sonuc["p50_gecikme_ms"],
            ardisik_sonuc["p90_gecikme_ms"],
            ardisik_sonuc["p99_gecikme_ms"]
        ]
        dinamik_lat = [
            dinamik_sonuc["ortalama_gecikme_ms"],
            dinamik_sonuc["p50_gecikme_ms"],
            dinamik_sonuc["p90_gecikme_ms"],
            dinamik_sonuc["p99_gecikme_ms"]
        ]

        x_k = np.arange(len(metrikler))
        w = 0.35

        ax3.bar(x_k - w/2, ardisik_lat, width=w, color="#e53e3e", label="Ardışık (B=1)")
        ax3.bar(x_k + w/2, dinamik_lat, width=w, color="#3182ce", label="Dinamik Batching")

        ax3.set_title("3. Uçtan Uca Gecikme Profili (Latency ms)", fontsize=12, fontweight="bold", color="#2b6cb0")
        ax3.set_xticks(x_k)
        ax3.set_xticklabels(metrikler, fontsize=9)
        ax3.set_ylabel("Gecikme (ms)", fontsize=10)
        ax3.legend(loc="upper left", frameon=True, fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 4: GPU Alt-Doğrusal Ölçeklenme (Sublinear Scaling)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        batch_boyutlari = batch_olcekleme_verisi["batch_boyutlari"]
        toplam_sureler_ms = batch_olcekleme_verisi["cikarim_sureleri_ms"]
        ornek_basina_sure_ms = [t / b for t, b in zip(toplam_sureler_ms, batch_boyutlari)]

        ax4.plot(batch_boyutlari, toplam_sureler_ms, marker="o", color="#d69e2e", linewidth=2.2, label="Toplam Batch Süresi (ms)")
        ax4.plot(batch_boyutlari, ornek_basina_sure_ms, marker="s", color="#805ad5", linewidth=2.2, linestyle="--", label="Örnek Başına Maliyet (ms/örnek)")

        ax4.set_title("4. GPU Tensör Çekirdeği Doygunluğu (Sublinear Scaling)", fontsize=12, fontweight="bold", color="#744210")
        ax4.set_xlabel("Batch Boyutu (B)", fontsize=10)
        ax4.set_ylabel("Süre (ms)", fontsize=10)
        ax4.legend(loc="upper right", frameon=True, fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 5: Toplam İşlem Süresi Kıyası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        sureler = [ardisik_sonuc["toplam_sure_sn"], dinamik_sonuc["toplam_sure_sn"]]

        bars5 = ax5.bar(modlar, sureler, color=["#e53e3e", "#38a169"], width=0.55, edgecolor="black")
        for bar, val in zip(bars5, sureler):
            ax5.text(bar.get_x() + bar.get_width()/2, val + max(sureler)*0.02, f"{val:.3f} sn", ha="center", fontsize=10, fontweight="bold")

        ax5.set_title("5. Toplam Yük Tamamlanma Süresi (200 İstek)", fontsize=12, fontweight="bold", color="#22543d")
        ax5.set_ylabel("Toplam Süre (Saniye)", fontsize=10)
        ax5.set_ylim(0, max(sureler) * 1.25)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        swot_metni = (
            "       DİNAMİK BATCHING ÇIKARIM SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • GPU işlem hacmini 10x-50x kat artırır (Maksimum Donanım Verimi).\n"
            "  • Asenkron Future API ile istemciye şeffaf sonuç iletimi.\n"
            "  • Sabit zaman aşımı (timeout) ile SLA kuyruk gecikmesini sınırlar.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Çok düşük trafik altında kuyrukta max_bekleme_ms kadar ek gecikme.\n"
            "  • Değişken boyutlu girdilerde (metin/LLM) padding maliyeti oluşur.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Triton Inference Server / vLLM / TensorRT-LLM entegrasyonu.\n"
            "  • Sürekli (Continuous/Iteration-level) batching ile LLM hızlandırma.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Max batch size çok büyük seçilirse GPU OOM (Out of Memory) riski."
        )

        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Dinamik Batching SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
