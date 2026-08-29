"""
Distributed Data Parallel (DDP) Teşhis Panosu Görselleştirici Modülü (Day 181 - FAZ 10).
6 panelli Ring All-Reduce İletişim Mimarisi, Gradient Bucketing, Çoklu GPU Ölçeklenme ve FAZ 10 Özet Kartı.
"""

import os
import sys
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class DDPGorsellestirici:
    """Distributed Data Parallel (DDP) Dağıtık Eğitim Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        olcek_raporu: Dict[str, Any],
        bucket_raporu: Optional[Dict[str, Any]] = None,
        egitim_logu: Optional[Dict[str, Any]] = None,
        kayit_yolu: str = "ciktilar/distributed_data_parallel_ddp_paneli.png",
    ):
        """6 panelli DDP dağıtık eğitim teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 181 (FAZ 10 BAŞLANGICI): PyTorch Distributed Data Parallel (DDP) — Ring All-Reduce & Gradient Bucketing",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Ring All-Reduce vs Parameter Server İletişim Hacmi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        gpu_counts = np.array([2, 4, 8, 16, 32, 64, 128])
        M_mb = 100.0  # 100 MB model parametresi

        # Parameter Server: 2 * (N - 1) * M
        ps_traffic = 2 * (gpu_counts - 1) * M_mb
        # Ring All-Reduce: 2 * (N - 1)/N * M
        ring_traffic = 2 * ((gpu_counts - 1) / gpu_counts) * M_mb

        ax1.plot(gpu_counts, ps_traffic, marker="o", color="#e74a3b", linewidth=2.5, label="Parameter Server (O(N) - Tıkanma)")
        ax1.plot(gpu_counts, ring_traffic, marker="s", color="#1cc88a", linewidth=2.5, label="Ring All-Reduce (O(1) - Sabit)")

        ax1.set_title("1. İletişim Hacmi Kıyası: Ring vs Parameter Server", fontsize=12, fontweight="bold")
        ax1.set_xlabel("GPU Rank Sayısı (N)")
        ax1.set_ylabel("Merkezi İletişim Yükü (MB - Log Skala)")
        ax1.set_yscale("log")
        ax1.set_xscale("log", base=2)
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Gradient Bucketing Öncesi vs Sonrası IPC Çağrıları
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kategoriler = ["Ayrı Tensörler (No-Bucket)", "DDP Gradient Bucketing (25MB)"]
        cagri_sayilari = [160, 3]
        gecikme_ms = [48.0, 4.2]

        ax2_twin = ax2.twinx()
        b1 = ax2.bar(kategoriler, cagri_sayilari, color="#4e73df", alpha=0.7, width=0.4, label="All-Reduce Çağrı Sayısı")
        l1 = ax2_twin.plot(kategoriler, gecikme_ms, color="#e74a3b", marker="o", linewidth=2.5, label="IPC Gecikmesi (ms)")

        ax2.set_title("2. Gradient Bucketing: IPC ve Gecikme Tasarrufu", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Sistem Çağrısı (Adet)", color="#4e73df")
        ax2_twin.set_ylabel("İletişim Gecikmesi (ms)", color="#e74a3b")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in b1:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.05, f"{int(yval)} çağrı", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 3: Çoklu GPU Ölçeklenebilirlik (Scaling Efficiency)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        olcek_verileri = olcek_raporu["karsilastirma"]
        gpus = [item["gpu_sayisi"] for item in olcek_verileri]
        hizlar = [item["hiz_imgs_per_sec"] for item in olcek_verileri]
        ideal_hizlar = [item["ideal_hiz"] for item in olcek_verileri]

        ax3.plot(gpus, ideal_hizlar, "--", color="#6c757d", linewidth=2.0, label="İdeal Lineer Hızlanma (%100)")
        ax3.plot(gpus, hizlar, marker="D", color="#36b9cc", linewidth=2.5, label="Gerçek DDP Hızı (İşlenen Örnek/sn)")

        ax3.set_title("3. DDP Çoklu GPU Ölçeklenebilirlik Verimi", fontsize=12, fontweight="bold")
        ax3.set_xlabel("GPU Sayısı")
        ax3.set_ylabel("İşlenen Örnek / Saniye (Throughput)")
        ax3.set_xscale("log", base=2)
        ax3.set_yscale("log")
        ax3.legend(loc="upper left", frameon=True)
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Overlapping Computation & Communication Çizelgesi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        gorevler = [
            ("Backward: Katman 3", 0, 15, "#4e73df"),
            ("All-Reduce: Bucket 3", 15, 30, "#1cc88a"),
            ("Backward: Katman 2", 15, 32, "#4e73df"),
            ("All-Reduce: Bucket 2", 32, 47, "#1cc88a"),
            ("Backward: Katman 1", 32, 50, "#4e73df"),
            ("All-Reduce: Bucket 1", 50, 65, "#1cc88a"),
            ("Optimizer Step", 65, 75, "#f6c23e"),
        ]

        y_labels = ["Katman 3", "Bucket 3 (Net)", "Katman 2", "Bucket 2 (Net)", "Katman 1", "Bucket 1 (Net)", "Optimizer"]
        y_pos = np.arange(len(gorevler))

        for i, (name, start, end, color) in enumerate(gorevler):
            ax4.barh(i, end - start, left=start, color=color, edgecolor="black", height=0.6)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(y_labels)
        ax4.invert_yaxis()
        ax4.set_title("4. Eşzamanlı Geri Geçiş & İletişim (Overlapping)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Zaman (Zaman Damgası ms)")
        ax4.grid(axis="x", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: DDP ve Ring All-Reduce İcra İzi Logu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. DDP Dağıtık Eğitim & Ring İcra İzi", fontsize=12, fontweight="bold", pad=10)

        log_metni = (
            "====================================================\n"
            "       PYTORCH DDP DISTRIBUTED TRAINING LOG         \n"
            "====================================================\n"
            f"MODEL ADI        : {olcek_raporu.get('model', 'Distributed Deep Model')}\n"
            f"İLETİŞİM TOPOLOJİ: Ring All-Reduce (Scatter-Reduce + All-Gather)\n"
            "----------------------------------------------------\n"
            "DDP DAĞITIK EĞİTİM ADIMLARI:\n"
            "1. DistributedSampler : Veriyi rank'lere çakışmasız bölme\n"
            "2. Forward Pass       : Her GPU yerel mini-batch ile kayıp bulur\n"
            "3. Backward Pass      : Geri geçiş başladığı an hook'lar tetiklenir\n"
            "4. Gradient Bucketing : 25 MB havuzlar doldukça All-Reduce başlar\n"
            "5. Ring Senkronizasyon: 2*(N-1) adımda tüm gradyanlar eşitlenir\n"
            "6. Optimizer Step     : Tüm rankler tamamen aynı ağırlıkta kalır\n"
            "----------------------------------------------------\n"
            f"ORTALAMA VERİM   : {olcek_raporu.get('ortalama_olcek_verimi', '%93.5')}\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, log_metni,
            fontsize=7.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 181 & FAZ 10 ÖZET KARTI
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 181 & FAZ 10 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 181 SUMMARY: DISTRIBUTED DATA PARALLEL (DDP) \n"
            "====================================================\n"
            "• Modül              : FAZ 10 (Ultra-MLOps & Triton)\n"
            "• Temel Algoritma    : Ring All-Reduce & Bucketing\n"
            "• İletişim Karmaşığı : O(1) sabit GPU başına yük\n"
            "• Verimlilik Kazancı : %93+ Lineer Ölçeklenebilirlik\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Parameter Server darboğazını kıran halka (Ring) topolojisi\n"
            "  2. Gradient Bucketing ile binlerce küçük IPC çağrısını engelleme\n"
            "  3. Geri geçiş (backward) ile iletişimi eşzamanlı çakıştırma\n"
            "  4. DistributedSampler ile deterministik veri partisyonlama\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 182 (Fully Sharded Data Parallel - FSDP)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ DDP Dağıtık Eğitim Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
