"""
Kubernetes KEDA & HPA 6 Panelli Görselleştirici Modülü (Day 197 - FAZ 10).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class KedaGorsellestirici:
    """Kubernetes KEDA 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        simulasyon_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/k8s_keda_paneli.png",
    ):
        """6 Panelli K8s KEDA Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 197: KUBERNETES KEDA & HPA İLE GPU KULLANIMINA GÖRE vLLM PODLARINI OTOMATİK ÖLÇEKLEME",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        adimlari = simulasyon_raporu["adimlari"]
        zaman_adlar = [f"{a['zaman_araligi'].split(' - ')[0]}\n({a['faz_tanimi']})" for a in adimlari]

        # -------------------------------------------------------------
        # PANEL 1: KEDA Mimari ve Metrik Akışı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        bilesenler = [
            "1. vLLM /metrics Uç Noktası",
            "2. Prometheus Metrik Kazıyıcı",
            "3. KEDA Custom Metrics Server",
            "4. K8s Horizontal Pod Autoscaler",
            "5. vLLM Worker GPU Podları",
        ]
        onem_skor = [1.0, 1.2, 1.5, 1.8, 2.0]
        ax1.barh(bilesenler, onem_skor, color=["#38bdf8", "#6366f1", "#8b5cf6", "#f59e0b", "#10b981"], height=0.5)
        ax1.set_xlabel("Sistem Katmanı Hiyerarşisi", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. KEDA Event-Driven Ölçekleme Mimarisi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: 24 Saatlik vLLM Pod Sayısı Değişimi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        podlar = [a["aktif_pod_sayisi"] for a in adimlari]
        ax2.plot(zaman_adlar, podlar, marker="o", color="#10b981", linewidth=2.5, markersize=8, label="Aktif vLLM Pod")
        ax2.set_ylabel("Aktif Pod / GPU Sayısı", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. 24 Saatlik Dinamik Pod Eğrisi (1 -> 8 -> 1)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.3)

        for i, p in enumerate(podlar):
            ax2.text(i, p + 0.35, f"{p} Pod", ha="center", color="#ffffff", fontweight="bold", fontsize=9)

        # -------------------------------------------------------------
        # PANEL 3: vLLM Kuyruk Derinliği ve KV Cache Doluluğu
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        x_pos = np.arange(len(zaman_adlar))
        w = 0.35
        bekleyenler = [a["kuyrukta_bekleyen"] for a in adimlari]
        kv_doluluk_vals = [float(a["kv_cache_doluluk"].replace("%", "")) for a in adimlari]

        ax3.bar(x_pos - w / 2.0, bekleyenler, width=w, color="#f59e0b", label="Bekleyen İstek Sayısı")
        ax3_twin = ax3.twinx()
        ax3_twin.plot(x_pos, kv_doluluk_vals, color="#ef4444", marker="s", linewidth=2.0, label="KV Cache Doluluk (%)")
        ax3_twin.set_ylabel("KV Cache Doluluk (%)", color="#ef4444", fontsize=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(zaman_adlar, fontsize=8)
        ax3.set_ylabel("Kuyruk Derinliği", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Tetikleyici Özel Metrikler (Queue & KV)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.grid(axis="y", linestyle=":", alpha=0.4)

        # -------------------------------------------------------------
        # PANEL 4: GPU Altyapı Maliyet Karşılaştırması
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        statik_saat = simulasyon_raporu["statik_gpu_saat"]
        dinamik_saat = simulasyon_raporu["dinamik_gpu_saat"]
        maliyet_bars = ax4.bar(["Statik Tahsis\n(Sabit 8 GPU)", "KEDA Dinamik\n(Otomatik Ölçekli)"], [statik_saat, dinamik_saat], color=["#ef4444", "#10b981"], width=0.45)
        ax4.set_ylabel("Günlük Toplam GPU-Saat", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Altyapı Maliyet Tasarrufu (%54+ Kazanç)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.4)

        for b in maliyet_bars:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 3.0, f"{h:.1f} GPU-Saat", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Ölçekleme Kararlılığı ve Cooldown Politikası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        politikalar = ["Scale-Up Gecikmesi", "Scale-Down Cooldown", "Health Probe Aralığı"]
        sureler = [5.0, 300.0, 10.0]
        bars5 = ax5.bar(politikalar, sureler, color=["#10b981", "#3b82f6", "#f59e0b"], width=0.45)
        ax5.set_yscale("log")
        ax5.set_ylabel("Süre (Saniye - Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. HPA Kararlılık ve Titreme (Flapping) Koruması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.4)

        for b, s in zip(bars5, sureler):
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h * 1.15, f"{int(s)}s", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 197 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 197: K8S KEDA vLLM AUTOSCALING KARNE\n"
            "----------------------------------------------------\n"
            "• Orkestrasyon Modu  : Kubernetes HPA + KEDA ScaledObject\n"
            "• Tetikleyici Metrik  : num_requests_waiting & gpu_cache_usage\n"
            "• Ölçekleme Aralığı   : 1 Min Pod -> 10 Max Pod\n"
            "• Scale-Up Hızı       : Anlık (Instant Reaction < 5s)\n"
            "• Scale-Down Koruması : 300 Saniye Soğuma (Zero Flapping)\n"
            "• Maliyet Tasarrufu   : %54.2 Günlük GPU Altyapı Kazancı\n"
            "----------------------------------------------------\n"
            "SONUÇ: Kubernetes üzerinde GPU kaynaklarını körü körüne\n"
            "tahsis etmek yerine, LLM kuyruk metrikleriyle akıllı ölçekleme!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=10,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
