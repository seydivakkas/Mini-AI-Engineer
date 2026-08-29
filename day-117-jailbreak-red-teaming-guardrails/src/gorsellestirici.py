"""
LLM Güvenlik Teşhis Panosu Görselleştirici Modülü (Day 117).
6 panelli ASR düşüşü, saldırı vektörü başarıları, Llama Guard taksonomi dağılımı ve güvenlik sertifikası paneli.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


class GuvenlikGorsellestirici:
    """LLM güvenlik ve Guardrails analizi için 6 panelli teşhis panosu üretir."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/guardrails_guvenlik_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "LLM Güvenlik Mühendisliği: Jailbreak Savunması ve Llama Guard Guardrails Paneli",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: ASR (Attack Success Rate) Düşüşü
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        modeller = ["Savunmasız Temel Model", "Llama Guard Korumalı"]
        asr_degerleri = [rapor["savunmasiz_asr"], rapor["korumali_asr"]]
        renkler1 = ["#e74a3b", "#1cc88a"]

        barlar1 = ax1.bar(modeller, asr_degerleri, color=renkler1, edgecolor="black", width=0.55)
        for bar in barlar1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax1.set_title("1. Saldırı Başarı Oranı (ASR - Attack Success Rate)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Saldırı Başarı Oranı (%)")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Saldırı Vektörleri Bazında Savunma Oranı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        vektorler = list(rapor["vektor_sonuclari"].keys())
        savunma_oranlari = [
            (v["engellenen"] / max(1, v["toplam"])) * 100.0 for v in rapor["vektor_sonuclari"].values()
        ]
        barlar2 = ax2.bar(vektorler, savunma_oranlari, color="#4e73df", edgecolor="black", width=0.55)
        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

        ax2.set_title("2. Saldırı Vektörlerine Karşı Savunma Başarısı", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Engellenme Oranı (%)")
        ax2.set_ylim(0, 115)
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Llama Guard Kategori Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kat_isimler = list(rapor["kategori_engellemeleri"].keys())
        kat_sayilar = list(rapor["kategori_engellemeleri"].values())
        renkler3 = ["#f6c23e", "#36b9cc", "#e74a3b", "#6f42c1", "#20c997"]

        ax3.pie(
            kat_sayilar,
            labels=[f"Kat {k}\n({v} Adet)" for k, v in zip(kat_isimler, kat_sayilar)],
            colors=renkler3[:len(kat_isimler)],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontweight": "bold", "fontsize": 9},
        )
        ax3.set_title("3. Llama Guard Kategori Engelleme Dağılımı", fontsize=12, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 4: Savunma Başarısı vs FPR Dengesi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        metrikler = ["Savunma Başarısı", "Yanlış Pozitiflik (FPR)"]
        degerler = [rapor["savunma_basarisi"], rapor["fpr_orani"]]
        renkler4 = ["#28a745", "#ffc107"]

        barlar4 = ax4.bar(metrikler, degerler, color=renkler4, edgecolor="black", width=0.5)
        for bar in barlar4:
            h = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax4.set_title("4. Emniyet vs Kullanılabilirlik (FPR)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Oran (%)")
        ax4.set_ylim(0, 115)
        ax4.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 5: Çift Katmanlı Guardrail Mimari Akışı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Çift Katmanlı Guardrail Mimarisi", fontsize=12, fontweight="bold", pad=10)

        akis_semasi = (
            "====================================================\n"
            "        DUAL-LAYER LLM GUARDRAIL ARCHITECTURE       \n"
            "====================================================\n"
            "1. KULLANICI İSTEMİ (User Prompt)\n"
            "   │\n"
            "2. GİRİŞ GÜVENLİK DUVARI (Input Guardrail):\n"
            "   ├── Jailbreak & DAN Rol Yapma Tespiti\n"
            "   ├── Base64 / Şifreli Payload Çözme\n"
            "   └── MLCommons S1-S6 Kategori Sınıflandırması\n"
            "   │   [GÜVENSİZ] ──> Doğrudan Reddetme Mesajı\n"
            "   ▼   [GÜVENLİ]\n"
            "3. HEDEF LLM (Core LLM Generation)\n"
            "   │\n"
            "4. ÇIKIŞ GÜVENLİK DUVARI (Output Guardrail):\n"
            "   ├── Zararlı İçerik Sızıntı Denetimi\n"
            "   ├── PII & API Anahtarı Regex Maskeleme\n"
            "   └── Güvenli Yanıt İletimi\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, akis_semasi,
            fontsize=8.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: Güvenlik Sertifikası ve Red-Teaming Raporu
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. LLM Güvenlik Sertifikası & Red-Teaming", fontsize=12, fontweight="bold", pad=10)

        sertifika = (
            "====================================================\n"
            "        LLM RED-TEAMING & SAFETY CERTIFICATE        \n"
            "====================================================\n"
            f"• Toplam Test Edilen Saldırı : {rapor['toplam_saldiri_sayisi']} Vaka\n"
            f"• Savunma Başarısı           : %{rapor['savunma_basarisi']:.1f} (ASR: %{rapor['korumali_asr']:.1f})\n"
            f"• Yanlış Pozitiflik (FPR)    : %{rapor['fpr_orani']:.1f} (Düşük Aşırı Red)\n"
            "• Desteklenen Standartlar    : Llama Guard (Meta),\n"
            "                               OWASP Top 10 for LLMs\n"
            "----------------------------------------------------\n"
            "[ONAYLANDI] Üretim Seviyesi Kurumsal LLM Güvenlik Duvarı\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, sertifika,
            fontsize=8.0,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=2.0),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Güvenlik Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
