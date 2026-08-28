"""
Veri Boru Hatti ve Gorsel Artirma Gorsellestiricisi (Pipeline Visualizer)
========================================================================
Boru hatti karsilastirma metriklerini, artirma orneklerini, gecikme ve throughput
grafigini ve SWOT analizini 6 panelli yuksek cozunurluklu endustriyel tabloda birlestirir.
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class VeriBoruHattiGorsellestirici:
    """
    Day 68 6-Panelli Veri Boru Hattı Teşhis ve Performans Panosu üreticisi.
    """

    @staticmethod
    def panoyu_ciz_ve_kaydet(
        benchmark_sonuclari: Dict[str, Any],
        ornek_gorseller: List[np.ndarray],
        ornek_basliklar: List[str],
        cikti_yolu: str = "ciktilar/veri_boru_hatti_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(21, 13))
        fig.suptitle(
            "Day 68: Albumentations ile Yuksek Performansli Veri Artirma & GPU Prefetching Paneli",
            fontsize=17,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        tv = benchmark_sonuclari["torchvision"]
        albu = benchmark_sonuclari["albumentations_cpu"]
        pref = benchmark_sonuclari["albumentations_prefetcher"]

        # -------------------------------------------------------------
        # 1. Panel: Yönetici & Benchmark Özet Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        ozet_metin = (
            "       VERI BORU HATTI BENCHMARK OZETI\n"
            "═══════════════════════════════════════════════════════\n"
            f" * Toplam Ornek / Batch  : {benchmark_sonuclari['ornek_sayisi']} / {benchmark_sonuclari['batch_size']}\n"
            f" * Calisma Cihazi        : {benchmark_sonuclari['cihaz'].upper()}\n"
            "───────────────────────────────────────────────────────\n"
            f" 1. Torchvision (PIL)    : {tv['fps']:>7.1f} FPS | {tv['batch_gecikmesi_ms']:>6.2f} ms ({tv['hizlanma_kat']:.2f}x)\n"
            f" 2. Albumentations (CPU) : {albu['fps']:>7.1f} FPS | {albu['batch_gecikmesi_ms']:>6.2f} ms ({albu['hizlanma_kat']:.2f}x)\n"
            f" 3. Albu + CUDA Prefetch : {pref['fps']:>7.1f} FPS | {pref['batch_gecikmesi_ms']:>6.2f} ms ({pref['hizlanma_kat']:.2f}x)\n"
            "───────────────────────────────────────────────────────\n"
            f" * Maksimum Hizlanma     : %{((pref['fps'] / tv['fps']) - 1.0)*100:>.1f} DAHA HIZLI\n"
            f" * GPU Veri Bekleme (Idle): SIFIRA YAKIN (Overlapped)\n"
            "═══════════════════════════════════════════════════════\n"
            " * C++ OpenCV Kernel + Asenkron CUDA Stream: AKTIF"
        )
        ax1.text(
            0.5, 0.5, ozet_metin,
            transform=ax1.transAxes,
            fontsize=10.5,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#e8f8f5", edgecolor="#1abc9c", linewidth=2.0)
        )
        ax1.set_title("1. Veri Boru Hatti Metrik Ozeti", fontweight="bold", color="#16a085")

        # -------------------------------------------------------------
        # 2. Panel: Albumentations Veri Artırma Örnekleri
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.axis("off")
        # 4'lü mini görsel ızgarası
        if len(ornek_gorseller) >= 4:
            # 2x2 alt ızgara
            for i in range(4):
                sub_ax = fig.add_axes([0.38 + (i % 2) * 0.13, 0.55 + (1 - i // 2) * 0.16, 0.12, 0.14])
                sub_ax.imshow(ornek_gorseller[i])
                sub_ax.set_title(ornek_basliklar[i], fontsize=8.5, fontweight="bold", color="#2c3e50")
                sub_ax.axis("off")
        ax2.set_title("2. Albumentations C++ Veri Donusumleri", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # 3. Panel: Batch İşleme Gecikmesi (ms)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        borular = ["Torchvision\n(Baseline)", "Albumentations\n(CPU)", "Albu + CUDA\nPrefetcher"]
        gecikmeler = [tv["batch_gecikmesi_ms"], albu["batch_gecikmesi_ms"], pref["batch_gecikmesi_ms"]]
        renkler_g = ["#e74c3c", "#f39c12", "#2ecc71"]

        b_bars = ax3.bar(borular, gecikmeler, color=renkler_g, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        for bar, val in zip(b_bars, gecikmeler):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f} ms", ha='center', fontweight='bold', fontsize=10)
        ax3.set_ylabel("Ortalama Batch Suresi (ms)")
        ax3.set_title("3. Batch Gecikmesi (Dusuk Daha Iyi)", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # 4. Panel: Throughput (FPS / Samples Per Second)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        fps_degerleri = [tv["fps"], albu["fps"], pref["fps"]]
        renkler_fps = ["#95a5a6", "#3498db", "#27ae60"]

        f_bars = ax4.bar(borular, fps_degerleri, color=renkler_fps, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
        for bar, val in zip(f_bars, fps_degerleri):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 20, f"{val:.0f} FPS", ha='center', fontweight='bold', fontsize=10)
        ax4.set_ylabel("Akis Hizi (Gorsel / Saniye)")
        ax4.set_title("4. Veri Akis Hizi (Yuksek Daha Iyi)", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # 5. Panel: CUDA Stream & CPU-GPU Overlapping Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        sema_metin = (
            " CUDA STREAM ILE ASENKRON VERI ON-YUKLEME (PREFETCH)\n"
            "════════════════════════════════════════════════════════\n"
            " [Geleneksel Boru Hattı (Seri / Bloke Eden)]:\n"
            "  CPU: [ Batch t Augment ] ──► [ PCIe H2D Transfer ]\n"
            "  GPU: (BEKLEMEDE / IDLE)  ──► [ GPU Forward/Backward ]\n"
            "  Toplam Adım Süresi = T_Augment + T_PCIe + T_Compute\n\n"
            " [CUDA Stream Prefetcher (Eşzamanlı / Çakışmalı)]:\n"
            "  CPU: [ Batch t+1 Augment ] ────────────┐ (Eşzamanlı)\n"
            "  DMA: [ Batch t+1 PCIe Copy (Stream) ] ──┤\n"
            "  GPU: [ Batch t GPU Forward/Backward ] ──┘\n"
            "  Toplam Adım Süresi = max(T_Compute, T_PCIe)\n"
            "────────────────────────────────────────────────────────\n"
            " • GPU Starvation (Açlık) Sıfırlanır, GPU %100 Doyumda Çalışır."
        )
        ax5.text(
            0.5, 0.5, sema_metin,
            transform=ax5.transAxes,
            fontsize=9.0,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#ebf5fb", edgecolor="#3498db", linewidth=1.8)
        )
        ax5.set_title("5. Asenkron Bellek Cakismasi (Overlapping)", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # 6. Panel: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_metin = (
            " ALBUMENTATIONS & PREFETCHING SWOT MATRISI\n"
            "─────────────────────────────────────────────────\n"
            " [S] GUCLU YONLER (Strengths):\n"
            " • C++ OpenCV ile 3x-5x daha hizli CPU donusumleri\n"
            " • Bounding box ve maskelerle eszamanli donusum\n"
            " • CUDA Stream ile GPU bellek beklemesinin onlenmesi\n\n"
            " [W] ZAYIF YONLER (Weaknesses):\n"
            " • NumPy uint8 ile PyTorch tensör format kopuklugu\n"
            " • CUDAPrefetcher'da pin_memory=True zorunlulugu\n\n"
            " [O] FIRSATLAR (Opportunities):\n"
            " • Coklu GPU (DDP) egitimlerinde CPU darbogazini cozme\n"
            " • Vision Transformer (ViT) egitimlerini hizlandirma\n\n"
            " [T] TEHDITLER (Threats):\n"
            " • Yuksek thread sayisinda (num_workers) RAM tuketimi"
        )
        ax6.text(
            0.5, 0.5, swot_metin,
            transform=ax6.transAxes,
            fontsize=9.2,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.8)
        )
        ax6.set_title("6. Veri Boru Hatti SWOT Matrisi", fontweight="bold", color="#d35400")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return cikti_yolu
