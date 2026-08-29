"""
Model Context Protocol (MCP) 6 Panelli Görselleştirici Modülü (Day 221 - FAZ 12).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class MCPGorsellestirici:
    """MCP 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/mcp_protokol_paneli.png",
    ):
        """6 Panelli MCP Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 221 (FAZ 12 BAŞLANGICI): MODEL CONTEXT PROTOCOL (MCP) - STANDART ARAÇ SUNUCUSU VE İSTEMCİSİ",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["Özel Ad-Hoc\nYapıştırıcılar", "Standart Model Context\nProtocol (MCP)"]

        # -------------------------------------------------------------
        # PANEL 1: MCP İstemci-Sunucu Akışı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asamalar = ["1. LLM Host (Claude/Antigravity)", "2. JSON-RPC 2.0 İstemci", "3. tools/list Dinamik Keşif", "4. Schema Doğrulama", "5. tools/call Yürütme"]
        onemler = [1.0, 1.5, 1.9, 2.3, 2.7]
        ax1.barh(asamalar[::-1], onemler[::-1], color=["#38bdf8", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899"], height=0.45)
        ax1.set_xlabel("İşlem Aşamaları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. MCP İstemci-Sunucu İletişim Hattı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Entegrasyon Süresi (Saat)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        sureler = [
            karsilastirma["entegrasyon_suresi_saat"]["Ozel_Ad_Hoc_Yapistiricilar"],
            karsilastirma["entegrasyon_suresi_saat"]["Standart_MCP_Protokolu"],
        ]
        bars2 = ax2.bar(modeller, sureler, color=["#ef4444", "#10b981"], width=0.4)
        ax2.set_ylabel("Entegrasyon Süresi (Saat)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Entegrasyon Hızı (336h -> 2h)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 390)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 8.0, f"{h:.1f} Saat", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Birlikte Çalışabilirlik (Interoperability %)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        birlikte = [
            karsilastirma["birlikte_calisabilirlik_yuzdesi"]["Ozel_Ad_Hoc_Yapistiricilar"],
            karsilastirma["birlikte_calisabilirlik_yuzdesi"]["Standart_MCP_Protokolu"],
        ]
        bars3 = ax3.bar(modeller, birlikte, color=["#ef4444", "#10b981"], width=0.4)
        ax3.set_ylabel("Uyumluluk Oranı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Ekosistem Uyumluluğu (%20 -> %100)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 120)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Çalışma Zamanı Hata Oranı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        hata = [
            karsilastirma["calisma_zamani_hata_orani"]["Ozel_Ad_Hoc_Yapistiricilar"],
            karsilastirma["calisma_zamani_hata_orani"]["Standart_MCP_Protokolu"],
        ]
        bars4 = ax4.bar(modeller, hata, color=["#ef4444", "#10b981"], width=0.4)
        ax4.set_ylabel("Hata Oranı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Çalışma Zamanı Kararlılığı (%12.5 -> %0.1)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 16)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 0.3, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Bakım Eforu Puanı (/10)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        bakim = [
            karsilastirma["bakim_eforu_skoru"]["Ozel_Ad_Hoc_Yapistiricilar"],
            karsilastirma["bakim_eforu_skoru"]["Standart_MCP_Protokolu"],
        ]
        bars5 = ax5.bar(modeller, bakim, color=["#ef4444", "#10b981"], width=0.4)
        ax5.set_ylabel("Bakım Yükü (1-10 Puan)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Geliştirici Bakım Yükü (8.5 -> 1.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 11)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 0.2, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 221 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 221: MODEL CONTEXT PROTOCOL (MCP) KARNESİ\n"
            "----------------------------------------------------\n"
            "• Standart            : Model Context Protocol (MCP - Anthropic)\n"
            "• İletişim Protokolü  : JSON-RPC 2.0 (stdio / SSE)\n"
            "• Temel Primitifler   : tools/list, tools/call, resources/read\n"
            "• Entegrasyon Hızı    : 336 Saat (14 Gün) -> 2 Saat\n"
            "• Ekosistem Uyumu     : %100 (Claude, Antigravity, VS Code)\n"
            "• Hata Oranı          : %12.5 -> %0.1 (Deterministik Şema)\n"
            "----------------------------------------------------\n"
            "SONUÇ: FAZ 12 Başladı! Ajanlarımız artık tüm harici\n"
            "araç ve veritabanlarına evrensel standartla bağlanıyor!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.5,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
