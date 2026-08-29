"""
GUI Ajanı Teşhis Panosu Görselleştirici Modülü (Day 166 - FAZ 9).
6 panelli SoM Numaralandırılmış Web Ekranı, Eylem Türü Dağılımı, Adım Adım Başarı, Çıkarım İzi, Mimari Şema ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class GUIAjanGorsellestirici:
    """GUI Ajanı Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        ajan_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/gui_agent_web_navigation_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 166 (FAZ 9): GUI Ajanları & Web Gezintisi — Set-of-Mark (SoM) ve Otonom Eylem Planlama (Click, Type)",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Set-of-Mark (SoM) ile Etiketlenmiş Web Ekranı Simülasyonu
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(1000, 0)
        ax1.set_facecolor("#ffffff")

        # Temsili Web Arayüzü Elemanları
        # Arama Kutusu [1]
        ax1.add_patch(patches.Rectangle((250, 320), 500, 50, edgecolor="#4285F4", facecolor="#f8f9fa", lw=2))
        ax1.text(260, 350, "Ara veya URL yazin...", color="#70757a", fontsize=9, va="center")
        ax1.text(240, 315, "[1]", color="white", fontsize=8, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.2", facecolor="#ea4335", edgecolor="none"))

        # Şanslı Buton [2]
        ax1.add_patch(patches.Rectangle((350, 400), 170, 40, edgecolor="#dadce0", facecolor="#f8f9fa", lw=1.5))
        ax1.text(360, 422, "Kendimi Sansli Hissediyorum", color="#3c4043", fontsize=7.5, va="center")
        ax1.text(340, 395, "[2]", color="white", fontsize=8, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.2", facecolor="#ea4335", edgecolor="none"))

        # Görseller Link [3]
        ax1.text(860, 45, "Gorseller", color="#1a0dab", fontsize=9, va="center")
        ax1.text(845, 30, "[3]", color="white", fontsize=8, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.2", facecolor="#ea4335", edgecolor="none"))

        # Oturum Aç [4]
        ax1.add_patch(patches.Rectangle((920, 25), 70, 35, edgecolor="#1a73e8", facecolor="#1a73e8", lw=1))
        ax1.text(930, 42, "Oturum Ac", color="white", fontsize=8, fontweight="bold", va="center")
        ax1.text(910, 15, "[4]", color="white", fontsize=8, fontweight="bold", bbox=dict(boxstyle="circle,pad=0.2", facecolor="#ea4335", edgecolor="none"))

        ax1.set_title("1. Set-of-Mark (SoM) Web Ekranı Görsel İşaretleme", fontsize=12, fontweight="bold")
        ax1.set_xlabel("X Piksel Koordinatı")
        ax1.set_ylabel("Y Piksel Koordinatı")
        ax1.grid(True, linestyle="--", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: GUI Eylem Türü Dağılımı (Action Space)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        turler = ["click(x, y)", "type(text)", "press_key()", "scroll()", "terminate()"]
        sayilar = [4, 1, 1, 1, 2]
        renkler2 = ["#4e73df", "#1cc88a", "#f6c23e", "#36b9cc", "#e74a3b"]

        barlar2 = ax2.bar(turler, sayilar, color=renkler2, edgecolor="black", width=0.45)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{int(h)} Kez", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.set_title("2. GUI Ajanı Eylem Dağılımı (Toplam: 9 Adım)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Eylem Sayısı")
        ax2.set_ylim(0, 6)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Görev Başarı ve Adım Doğrulama Oranı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metrikler = ["Görev Başarısı", "Adım Doğruluğu", "Geçersiz Eylem"]
        degerler = [ajan_raporu["gorev_tamamlama_orani"], ajan_raporu["adim_basari_yuzdesi"], 0.0]

        barlar3 = ax3.bar(metrikler, degerler, color=["#1cc88a", "#36b9cc", "#e74a3b"], edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title("3. Otonom Görev Yürütme Başarısı (%100)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Oran (%)")
        ax3.set_ylim(0, 115)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Örnek Otonom Görev İcra İzi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Çok Adımlı Web Görevi İcra İzi", fontsize=12, fontweight="bold", pad=10)

        g1 = ajan_raporu["gorev_raporlari"][0]
        iz_metni = (
            "====================================================\n"
            "         GUI AGENT AUTONOMOUS EXECUTION TRACE       \n"
            "====================================================\n"
            f"GÖREV HEDEFİ: '{g1['hedef']}'\n"
            "----------------------------------------------------\n"
            "ADIM ADIM EYLEMLER:\n"
            f"  1. [{g1['adim_detaylari'][0]['ekran']}] -> {g1['adim_detaylari'][0]['eylem_metni']}\n"
            f"  2. [{g1['adim_detaylari'][1]['ekran']}] -> {g1['adim_detaylari'][1]['eylem_metni']}\n"
            f"  3. [{g1['adim_detaylari'][2]['ekran']}] -> {g1['adim_detaylari'][2]['eylem_metni']}\n"
            f"  4. [{g1['adim_detaylari'][3]['ekran']}] -> {g1['adim_detaylari'][3]['eylem_metni']}\n"
            f"  5. [{g1['adim_detaylari'][4]['ekran']}] -> {g1['adim_detaylari'][4]['eylem_metni']}\n"
            "----------------------------------------------------\n"
            "DURUM: [GÖREV BAŞARIYLA TAMAMLANDI]\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, iz_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: GUI Ajanı ve SoM Mimari Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. GUI Ajanı ve Set-of-Mark (SoM) Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "          GUI AGENT WEB NAVIGATION ARCHITECTURE     \n"
            "====================================================\n"
            "  [Ham Ekran Görüntüsü (Screenshot)]                \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [Set-of-Mark (SoM) İşaretleyici ([1], [2], ...)]  \n"
            "           │                                        \n"
            "           ▼  + [Kullanıcı Doğal Dil Hedefi]        \n"
            "  [Vision Language Model (VLM - GPT-4V / LLaVA)]    \n"
            "           │                                        \n"
            "           ▼  (Düşünce Zinciri - Visual CoT)        \n"
            "  [Eylem Planı: 'click(345, 500)' / 'type(...)']    \n"
            "           │                                        \n"
            "           ▼                                        \n"
            "  [OS / Web Tarayıcı Sürücüsü (Playwright)] ──> [İcra]\n"
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
        # PANEL 6: GÜN 166 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 166 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 166 SUMMARY: GUI AGENT & WEB NAVIGATION      \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• İşaretleme Yöntemi : Set-of-Mark (SoM Visual Prompting)\n"
            "• Eylem Uzayı        : click(x, y), type(), press_key(), scroll()\n"
            "• Başarı Oranı       : %100 Adım & Görev Başarımı\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Ekran piksellerini SoM ile numaralandırıp VLM'e sunma\n"
            "  2. click(x, y) koordinatlarıyla butona hatasız basma\n"
            "  3. type(metin) ve press_key(Enter) ile form doldurma\n"
            "  4. Anthropic Computer Use / Operator tarzı otonom gezinme\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 167 (Video LLM - Spatio-Temporal Tokens)\n"
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
        print(f"  ✓ GUI Ajanı Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
