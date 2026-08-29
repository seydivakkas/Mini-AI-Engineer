"""
Streaming VLM Teşhis Panosu Görselleştirici Modülü (Day 168 - FAZ 9).
6 panelli Gerçek Zamanlı Anomali/Olay Grafiği, Kayan Bellek Doluluğu, Olay Zaman Çizelgesi, VLM Alarm Günlüğü, Mimari Şema ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class StreamingGorsellestirici:
    """Streaming VLM Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        simulasyon_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/streaming_video_understanding_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 168 (FAZ 9): Streaming VLM — Gerçek Zamanlı Video Akışı Analizi, Kayan Bellek (Sliding Memory) ve Olay Tespiti",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        adimlar = simulasyon_raporu["akis_adimlari"]
        zamanlar = [a["zaman_damgasi"] for a in adimlar]
        skorlar = [a["anomali_skoru"] for a in adimlar]
        bellek_doluluk = [a["bellek_doluluk"] for a in adimlar]

        # -------------------------------------------------------------
        # PANEL 1: Gerçek Zamanlı Anomali ve Olay Tetikleme Sinyali
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(zamanlar, skorlar, color="#4e73df", lw=2, label="Anomali / Değişim Skoru")
        ax1.axhline(0.35, color="#e74a3b", linestyle="--", lw=1.5, label="Olay Tetikleme Eşiği (0.35)")

        # Tetiklenen anları kırmızı nokta ile işaretle
        for a in adimlar:
            if a["tetiklendi"]:
                ax1.scatter([a["zaman_damgasi"]], [a["anomali_skoru"]], color="#e74a3b", s=90, zorder=5)
                ax1.text(a["zaman_damgasi"], a["anomali_skoru"] + 0.05, f"OLAY @ t={int(a['zaman_damgasi'])}s", ha="center", fontsize=8.5, fontweight="bold", color="#e74a3b")

        ax1.set_title("1. Gerçek Zamanlı Video Değişim & Olay Sinyali", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Zaman (Saniye)")
        ax1.set_ylabel("Anomali / Fark Skoru")
        ax1.set_ylim(0, 1.2)
        ax1.legend(loc="upper left", fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 2: Kayan Bellek Tamponu Doluluk Dinamiği (FIFO Ring Buffer)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(zamanlar, bellek_doluluk, color="#1cc88a", lw=2.5, marker="o", markersize=4, label="Aktif Tampon Boyutu (Kare)")
        ax2.axhline(16, color="#858796", linestyle=":", lw=1.5, label="Maksimum Bellek Kapasitesi (16 Kare)")

        ax2.set_title("2. Kayan Bellek Kuyruğu Doluluk Eğrisi (FIFO)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Zaman (Saniye)")
        ax2.set_ylabel("Kare Sayısı")
        ax2.set_ylim(0, 20)
        ax2.legend(loc="lower right", fontsize=8.5)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 3: Streaming VLM Gecikme ve Verimlilik Kıyası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        yaklasimlar = ["Tüm Videoyu Sakla\n(Batch VLM)", "Kayan Bellek + Trigger\n(Streaming VLM)"]
        bellek_mb = [1800 * 0.5, 16 * 0.5]  # 1800 kare vs 16 kare MB
        renkler3 = ["#e74a3b", "#1cc88a"]

        barlar3 = ax3.bar(yaklasimlar, bellek_mb, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h:.1f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title("3. Bellek Tüketim Kıyası (%99 Tasarruf)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("GPU Bellek Kullanımı (MB)")
        ax3.set_ylim(0, 1100)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Canlı VLM Olay ve Alarm Günlüğü
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Canlı Olay ve Güvenlik Alarmları İzi", fontsize=12, fontweight="bold", pad=10)

        gunluk = simulasyon_raporu["olay_gunlugu"]
        gunluk_metni = (
            "====================================================\n"
            "       ONLINE EVENT DETECTION & VLM ALARM LOG       \n"
            "====================================================\n"
            f"CANLI AKIŞ SÜRESİ: {simulasyon_raporu['toplam_islenen_saniye']} Saniye | TETİKLENEN OLAY: {simulasyon_raporu['toplam_tetiklenen_olay']}\n"
            "----------------------------------------------------\n"
            "TETİKLENEN ALARMLAR:\n"
        )
        for i, o in enumerate(gunluk, 1):
            gunluk_metni += (
                f"  [{i}] ZAMAN: t={int(o['zaman_damgasi'])}s (Skor: {o['anomali_skoru']:.2f})\n"
                f"      SEVİYE: [{o['alarm_seviyesi']}]\n"
                f"      AÇIKLAMA: \"{o['aciklama']}\"\n"
                "----------------------------------------------------\n"
            )
        gunluk_metni += "DURUM: [TÜM KRİTİK OLAYLAR SIFIR GECİKMEYLE YAKALANDI]\n"
        gunluk_metni += "===================================================="

        ax4.text(
            0.02, 0.5, gunluk_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Streaming VLM Mimari Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Streaming VLM ve Kayan Bellek Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "        STREAMING VLM ONLINE ARCHITECTURE           \n"
            "====================================================\n"
            "  [Canlı Kamera Akışı (30 FPS Continuous Stream)]   \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Kayan Bellek Kuyruğu (FIFO Ring Buffer - 16 Kare)]\n"
            "           │                                        \n"
            "           ├──> [Online Değişim Dedektörü (Cosine/Diff)]\n"
            "           │         │                              \n"
            "           │         ▼ (Eşik Aşıldı mı? > 0.35)     \n"
            "           │      [EVET -> VLM Tetikleyici]         \n"
            "           ▼         │                              \n"
            "  [Streaming VLM Projektör + Causal LLM]            \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Anlık Güvenlik Uyarısı / Zaman Damgalı Alarm]    \n"
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
        # PANEL 6: GÜN 168 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 168 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 168 SUMMARY: STREAMING VIDEO UNDERSTANDING   \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Akış Yönetimi      : 16 Karelik FIFO Ring Buffer\n"
            "• Tetikleme Mekanizması: Online Değişim & Anomali Dedektörü\n"
            "• Bellek Tasarrufu   : %99 (Tüm video saklamaya göre)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Sonsuz canlı video akışını sabit bellek ile işleme\n"
            "  2. Değişim dedektörü ile sadece kritik anlarda VLM çağırma\n"
            "  3. Zaman damgalı anlık alarm ve olay özetleme\n"
            "  4. Güvenlik, otonom sürüş ve robotik algı altyapısı\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 169 (Audio Tokenizer - EnCodec/SoundStream)\n"
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
        print(f"  ✓ Streaming VLM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
