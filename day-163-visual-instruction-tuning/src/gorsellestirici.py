"""
Görsel SFT Teşhis Panosu Görselleştirici Modülü (Day 163 - FAZ 9).
6 panelli Kayıp Maskeleme, SFT Kayıp Eğrisi, Komut Kategorileri Dağılımı, SFT Çıktı Örneği, Eğitim Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class VisualSFTGorsellestirici:
    """Visual Instruction Tuning Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        egitim_raporu: Dict[str, Any],
        ornek_veriler: List[Dict[str, Any]],
        kayit_yolu: str = "ciktilar/visual_instruction_tuning_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 163 (FAZ 9): Görsel Komut İnce Ayarı (Visual SFT) — Kayıp Maskeleme ve Çok Turlu Görsel Sohbet",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Token Kayıp Maskeleme Oranı (Görsel + Prompt vs Yanıt)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        kategoriler1 = ["Maskelenen Görsel Tokenlar (256)", "Maskelenen Kullanıcı Prompt (14)", "Eğitilen Asistan Yanıtı (18)"]
        sayilar1 = [256, 14, 18]
        renkler1 = ["#6c757d", "#e74a3b", "#1cc88a"]

        ax1.pie(sayilar1, labels=kategoriler1, autopct="%1.1f%%", colors=renkler1, startangle=140, explode=(0.05, 0.05, 0.1))
        ax1.set_title("1. Token Maskeleme Dağılımı (Kayıp = -100 vs Target)", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 2: Görsel SFT Eğitim Kayıp Eğrisi (Loss Curve)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        adimlar = list(range(1, len(egitim_raporu["kayip_gecmisi"]) + 1))
        kayiplar = egitim_raporu["kayip_gecmisi"]

        ax2.plot(adimlar, kayiplar, marker="o", lw=2.5, color="#4e73df", label="Visual SFT Cross-Entropy")
        for x, y in zip(adimlar, kayiplar):
            ax2.text(x, y + 0.05, f"{y:.3f}", ha="center", fontsize=9, fontweight="bold")

        ax2.set_title(f"2. Görsel SFT Kayıp Azalışı (%{egitim_raporu['kayip_dususu_yuzdesi']} İyileşme)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Eğitim Adımı")
        ax2.set_ylabel("Kayıp (Loss)")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper right")

        # -------------------------------------------------------------
        # PANEL 3: LLaVA-Instruct Veri Seti Kategori Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kat_isimleri = ["Kısa VQA\n(58k Örnek)", "Detaylı Açıklama\n(45k Örnek)", "Karmaşık Mantık\n(47k Örnek)"]
        oranlar3 = [58, 45, 47]
        renkler3 = ["#36b9cc", "#f6c23e", "#1cc88a"]

        barlar3 = ax3.bar(kat_isimleri, oranlar3, color=renkler3, edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{int(h)}k", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title("3. LLaVA-Instruct-150k Komut Dağılımı", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Örnek Sayısı (x1000)")
        ax3.set_ylim(0, 70)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Örnek Çok Turlu Görsel Komut Diyaloğu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Örnek Görsel Komut ve Yanıt Formatı", fontsize=12, fontweight="bold", pad=10)

        ornek = ornek_veriler[0]
        diyalog_metni = (
            "====================================================\n"
            "         VISUAL INSTRUCTION CONVERSATION            \n"
            "====================================================\n"
            f"KATEGORİ: [{ornek['kategori']}] | GÖRSEL: {ornek['goruntu_adi']}\n"
            "----------------------------------------------------\n"
            f"İNSAN:\n  {ornek['diyalog'][0]['metin']}\n\n"
            f"ASİSTAN (Eğitilen Hedef):\n  {ornek['diyalog'][1]['metin']}\n"
            "----------------------------------------------------\n"
            f"  • Görüntü Tokenları : 256 adet (Maskelendi: -100)\n"
            f"  • İnsan Promptu     : {ornek['prompt_token_uzunlugu']} token (Maskelendi: -100)\n"
            f"  • Asistan Yanıtı    : {ornek['yanit_token_uzunlugu']} token (Kayıp Hesaplandı)\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, diyalog_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Görsel SFT Eğitim ve Maskeleme Mimarisi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Target-Only Kayıp Maskeleme Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         VISUAL SFT LOSS MASKING ARCHITECTURE       \n"
            "====================================================\n"
            " GİRDİ DİZİSİ (Forward):                            \n"
            "  [IMG_1 ... IMG_256] [HUMAN_PROMPT] [ASSISTANT_RESP]\n"
            "           │                 │               │      \n"
            "           ▼                 ▼               ▼      \n"
            " HEDEF ETİKETLER (Labels):                          \n"
            "  [-100  ...  -100 ] [ -100 ... -100] [ T_1 ... T_K ]\n"
            "           │                 │               │      \n"
            "           ▼ (Yoksayılır)    ▼ (Yoksayılır)  ▼      \n"
            " KAYIP FONKSİYONU:                                  \n"
            "  CrossEntropy(Logits, Labels, ignore_index=-100)   \n"
            "  ==> Sadece Asistan Tokenlarında Gradyan Üretilir! \n"
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
        # PANEL 6: GÜN 163 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 163 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 163 SUMMARY: VISUAL INSTRUCTION TUNING       \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Eğitim Stratejisi  : Target-Only Loss Masking (-100)\n"
            "• Veri Seti Formatı  : LLaVA-Instruct (VQA + Desc + Reasoning)\n"
            "• Kayıp Azalışı      : %" + str(egitim_raporu["kayip_dususu_yuzdesi"]) + " İyileşme\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Resim ve prompt piksellerini kayıptan yoksayma (-100)\n"
            "  2. Modelin sadece asistan yanıtına odaklanmasını sağlama\n"
            "  3. Çok turlu görsel diyalog formatlama standartları\n"
            "  4. Uçtan uca Visual SFT eğitim döngüsü tamamlama\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 164 (Spatial Grounding & Bounding Box VLM)\n"
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
        print(f"  ✓ Görsel SFT Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
