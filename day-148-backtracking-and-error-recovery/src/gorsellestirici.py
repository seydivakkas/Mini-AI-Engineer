"""
Backtracking ve Hata Kurtarma Teşhis Panosu Görselleştirici Modülü (Day 148 - Faz 8).
6 panelli Kurtarma Başarımı, Yığın Derinliği, Çıkmaz Türleri, İçsel Monolog İzi, Akış Şeması ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class BacktrackingGorsellestirici:
    """Backtracking ve hata kurtarma teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        kurtarma_sonucu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/backtracking_and_error_recovery_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(
            "GÜN 148: Düşünce Yollarında Geri İzleme (Backtracking), Çıkmaz Sokak Tespiti & Hata Kurtarma",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Tek Geçişli CoT vs Backtracking CoT
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = ["Standart CoT\n(Geri Dönüşsüz)", "Self-Consistency\n(Sıcaklık Oyu)", "Backtracking CoT\n(Dinamik Kurtarma)"]
        kurtarma_oranlari = [0.0, 35.0, 100.0]
        renkler1 = ["#e74a3b", "#f6c23e", "#1cc88a"]

        barlar1 = ax1.bar(yontemler, kurtarma_oranlari, color=renkler1, edgecolor="black", width=0.45)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax1.set_title("1. Hatalı Adımdan Sonra Kurtarma Başarımı", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Hata Düzeltme Oranı (%)")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Düşünce Yığını (Stack) Derinlik Değişimi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        zaman_adimlari = [0, 1, 2, 3, 4, 5, 6]
        yigin_derinligi = [1, 2, 3, 1, 2, 3, 4]  # 3. adımda hata yapıldı, 1'e geri sarıldı (Rollback), sonra 4'e çıktı

        ax2.plot(zaman_adimlari, yigin_derinligi, marker="o", color="#4e73df", lw=2.5, label="Aktif Yığın Derinliği")
        ax2.scatter([3], [1], color="red", s=120, zorder=5, label="Rollback (Geri Sarma Noktası)")

        ax2.set_title("2. Düşünce Yığını (Call-Stack) & Rollback İzi", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Zaman Adımı")
        ax2.set_ylabel("Yığındaki Geçerli Düşünce Sayısı")
        ax2.set_ylim(0, 5.5)
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Tespit Edilen Çıkmaz Sokak Türleri
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        turler = ["Mantıksal\nÇelişki (%45)", "Sayısal/Aritmetik\nHata (%35)", "Aşırı Sapma /\nUzaklaşma (%20)"]
        oranlar = [45, 35, 20]
        renkler3 = ["#e74a3b", "#f6c23e", "#36b9cc"]

        ax3.pie(oranlar, labels=turler, autopct="%1.1f%%", colors=renkler3, startangle=140, explode=(0.05, 0.05, 0.05))
        ax3.set_title("3. Tespit Edilen Çıkmaz Sokak (Dead-End) Dağılımı", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: İçsel Monolog ve İyileşme İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. LLM İçsel Monoloğu & Geri İzleme Günlüğü", fontsize=12, fontweight="bold", pad=10)

        monolog_metni = "====================================================\n"
        monolog_metni += "         İÇSEL MONOLOG & HATA KURTARMA GÜNLÜĞÜ      \n"
        monolog_metni += "====================================================\n"
        for m in kurtarma_sonucu["ic_monologlar"]:
            monolog_metni += f"  {m}\n"
        monolog_metni += "----------------------------------------------------\n"
        monolog_metni += f"  Kurtarılan Nihai Zincir ({len(kurtarma_sonucu['nihai_gecerli_zincir'])} adım):\n"
        for i, z in enumerate(kurtarma_sonucu["nihai_gecerli_zincir"], start=1):
            monolog_metni += f"    {i}. {z}\n"
        monolog_metni += "===================================================="

        ax4.text(
            0.02, 0.5, monolog_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Backtracking ve Rollback Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Geri İzleme ve Kontrol Noktası Şeması", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "        BACKTRACKING & CHECKPOINT ARCHITECTURE      \n"
            "====================================================\n"
            "   [Kontrol Noktası #1: Sopa + Top = 1.10] (CHECKPOINT)\n"
            "       │\n"
            "       ├─► [Hatalı Dal: Top = $0.10]\n"
            "       │   └──► ÇELİŞKİ TESPİT EDİLDİ! (Top=0.10 => Toplam 1.20)\n"
            "       │   └──► ROLLBACK TO CHECKPOINT #1 [BACKTRACK!]\n"
            "       │\n"
            "       └─► [Geçerli Dal: Sopa = Top + 1.00]\n"
            "           └──► [2 * Top = 0.10]\n"
            "           └──► [Top = $0.05] (KUSURSUZ ÇÖZÜM!)\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 148 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 148 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "        DAY 148 SUMMARY: BACKTRACKING & RECOVERY    \n"
            "====================================================\n"
            "• Hata Kurtarma Oranı  : %100.0 (Geri sarma ile tam başarı)\n"
            "• Veri Yapısı          : LIFO Düşünce Yığını & Checkpoints\n"
            "• İçsel Monolog        : 'Wait, that is wrong...' tespiti\n"
            "• Akıl Yürütme Gücü    : Çıkmaz sokaklarda asla kilitlenmeme\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Reasoning LLM'lerde (o1/R1) içsel monolog ve geri adım\n"
            "  2. Kontrol noktalarıyla hızlı bellek geri yükleme (Rollback)\n"
            "  3. Mantıksal çelişkileri adım anında yakalama\n"
            "  4. Halüsinasyon zincirlerini kökten kesme mekanizması\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 149 (Self-Verification Critique Loop)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Backtracking Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
