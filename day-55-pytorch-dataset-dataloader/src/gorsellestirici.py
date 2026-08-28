"""
6-Panelli PyTorch DataLoader Performans ve Darboğaz Teşhis Panosu (DataLoader Profiler Dashboard).
"""

from typing import Dict, Any, List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoaderAnalizGorsellestirici:
    """DataLoader hızlanma, gecikme, worker ölçeklenme ve bellek sabitleme metriklerini 6 panelli panoda sunar."""

    @classmethod
    def panel_ciz(
        cls,
        benchmark_sonuclari: List[Dict[str, Any]],
        worker_tarama_sonuclari: List[Dict[str, Any]],
        hedef_path: str = "ciktilar/dataloader_darbogaz_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 55: İleri PyTorch DataLoader, num_workers, pin_memory ve Prefetch Darboğaz Optimizasyonu",
            fontsize=15, fontweight="bold", y=0.98
        )

        opt_cfg = benchmark_sonuclari[-1]
        base_cfg = benchmark_sonuclari[0]
        hizlanma = opt_cfg["hizlanma_carpani"]
        kart_renk = "#2ecc71" if hizlanma >= 2.0 else "#f39c12"

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özeti Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"DATALOADER OPTİMİZASYON YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Temel İşleme Hızı (Baseline) : {base_cfg['isleme_hizi_ornek_sn']} örnek/sn\n"
            f"• Optimize Hız (Production)    : {opt_cfg['isleme_hizi_ornek_sn']} örnek/sn\n"
            f"• Elde Edilen Hızlanma Çarpanı : {hizlanma:.2f}x HIZLANMA\n"
            f"• Sabitlenmiş Bellek (Pinned)  : AKTİF (Direct DMA)\n"
            f"• Kalıcı Süreçler (Persistent) : AKTİF (No Spawn Lag)\n"
            f"• Önceden Getirme (Prefetch)   : Factor = 2\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• GPU Darboğaz Durumu          :\n"
            f"  GPU Starvation %{base_cfg['gpu_starvation_orani']:.0f}'den %{opt_cfg['gpu_starvation_orani']:.0f}'ye düşürüldü!"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=kart_renk, alpha=0.22, edgecolor=kart_renk, linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. DataLoader Optimizasyon Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Veri İşleme Hızı (Throughput - Samples / Sec)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        isimler = [b["ad"].split(". ")[1] for b in benchmark_sonuclari]
        hizlar = [b["isleme_hizi_ornek_sn"] for b in benchmark_sonuclari]
        renkler = ["#95a5a6", "#3498db", "#9b59b6", "#2ecc71"]

        bars = ax2.bar(isimler, hizlar, color=renkler, edgecolor="black", width=0.55)
        ax2.set_ylabel("İşleme Hızı (Örnek / Saniye)")
        ax2.set_title("2. DataLoader İşleme Hızı (Throughput)", fontweight="bold", color="#2980b9")
        ax2.tick_params(axis="x", rotation=15)

        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + max(hizlar)*0.02, f"{int(h)}/s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 3: Toplam İşlem Süresi (Total Execution Time s)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        sureler = [b["toplam_sure_sn"] for b in benchmark_sonuclari]

        bars3 = ax3.bar(isimler, sureler, color=["#e74c3c", "#e67e22", "#f1c40f", "#27ae60"], edgecolor="black", width=0.55)
        ax3.set_ylabel("Toplam Süre (Saniye)")
        ax3.set_title("3. Toplam Batch İşleme Süresi", fontweight="bold", color="#8e44ad")
        ax3.tick_params(axis="x", rotation=15)

        for bar in bars3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., h + max(sureler)*0.02, f"{h:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 4: num_workers Ölçeklenme Eğrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        w_degerler = [w["num_workers"] for w in worker_tarama_sonuclari]
        w_hizlar = [w["isleme_hizi_ornek_sn"] for w in worker_tarama_sonuclari]

        ax4.plot(w_degerler, w_hizlar, marker="o", linewidth=2.5, markersize=8, color="#e67e22", label="Ölçülen Hız")
        ax4.set_xlabel("num_workers (Alt Süreç Sayısı)")
        ax4.set_ylabel("İşleme Hızı (Örnek / Saniye)")
        ax4.set_title("4. num_workers Ölçeklenme Eğrisi", fontweight="bold", color="#d35400")
        ax4.set_xticks(w_degerler)

        for x_val, y_val in zip(w_degerler, w_hizlar):
            ax4.text(x_val, y_val + max(w_hizlar)*0.03, f"{int(y_val)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 5: Bellek Transfer Süresi (Host -> GPU Transfer)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        tr_sureler = [b["ort_transfer_gecikmesi_ms"] for b in benchmark_sonuclari]

        bars5 = ax5.bar(isimler, tr_sureler, color=["#bdc3c7", "#34495e", "#16a085", "#2ecc71"], edgecolor="black", width=0.55)
        ax5.set_ylabel("Ortalama Transfer Süresi (ms)")
        ax5.set_title("5. Bellek Kopyalama & Transfer Gecikmesi", fontweight="bold", color="#c0392b")
        ax5.tick_params(axis="x", rotation=15)

        for bar in bars5:
            h = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., h + max(tr_sureler)*0.03, f"{h:.2f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 6: Hızlanma Çarpanı ve GPU Starvation
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        carpanlar = [b["hizlanma_carpani"] for b in benchmark_sonuclari]

        bars6 = ax6.bar(isimler, carpanlar, color=["#95a5a6", "#3498db", "#9b59b6", "#2ecc71"], edgecolor="black", width=0.55)
        ax6.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Baseline (1.0x)")
        ax6.set_ylabel("Hızlanma Çarpanı (x)")
        ax6.set_title("6. Baseline'a Göre Hızlanma Çarpanı", fontweight="bold", color="#27ae60")
        ax6.tick_params(axis="x", rotation=15)
        ax6.legend(loc="upper left")

        for bar in bars6:
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}x", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.36, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
