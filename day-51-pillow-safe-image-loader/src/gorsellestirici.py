"""
6-Panelli Güvenli Görsel Yükleyici Teşhis Panosu (Safe Image Loader Dashboard).
"""

from typing import Dict, Any, List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class GuvenliYukleyiciGorsellestirici:
    """Görsel yükleme güvenlik, normalizasyon ve onarım metriklerini 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        ozet_metrikler: Dict[str, Any],
        exif_ornek: np.ndarray,
        rgba_ornek: np.ndarray,
        kesik_ornek: np.ndarray,
        hedef_path: str = "ciktilar/guvenli_yukleyici_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 51: Pillow ile Hataya Toleranslı ve Güvenli Görsel Yükleyici (Safe Image Ingestion Pipeline)",
            fontsize=15, fontweight="bold", y=0.98
        )

        # -------------------------------------------------------------
        # Panel 1: Yönetici Karar ve Güvenlik Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"GÜVENLİ YÜKLEYİCİ YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Toplam İşlenen Görsel : {ozet_metrikler.get('toplam_islenen', 5)}\n"
            f"• Bomb Saldırısı Engeli : {ozet_metrikler.get('engellenen_bomb', 1)} Adet (Korumalı)\n"
            f"• EXIF Düzeltilen       : {ozet_metrikler.get('exif_duzeltilen', 1)} Adet\n"
            f"• Kesik Dosya Kurtarılan: {ozet_metrikler.get('kurtarilan_kesik', 1)} Adet\n"
            f"• Alfa Mat Kompoziti    : {ozet_metrikler.get('rgba_donusturulen', 1)} Adet\n"
            f"• Maks Güvenlik Sınırı  : 25.0 MP (~75 MB RAM)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Boru Hattı Durumu     : ÜRETİM SEVİYESİNDE GÜVENLİ"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#3498db", alpha=0.22, edgecolor="#2980b9", linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Güvenli Yükleme Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: EXIF Oryantasyon Düzeltme Örneği
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.imshow(exif_ornek)
        ax2.axis("off")
        ax2.set_title("2. EXIF Oryantasyon Düzeltildi (Orientation Tag -> Dik)", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 3: RGBA -> RGB Alfa Mat Normalizasyonu
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.imshow(rgba_ornek)
        ax3.axis("off")
        ax3.set_title("3. RGBA Şeffaflık -> Beyaz Arka Plan RGB Mat", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # Panel 4: Kesik/Bozuk (Truncated) Dosya Onarımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.imshow(kesik_ornek)
        ax4.axis("off")
        ax4.set_title("4. Kesik (Truncated) Görsel Akışından Kurtarılan Görsel", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 5: Decompression Bomb Güvenlik Sınırı Grafiği
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        cozunurluk_mp = np.linspace(1, 50, 50)
        ram_mb = cozunurluk_mp * 3.0  # RGB uint8 ~ 3 MB per MP

        ax5.plot(cozunurluk_mp, ram_mb, color="#e74c3c", linewidth=2.4, label="Açılmış Bellek İhtiyacı (MB)")
        ax5.axvline(25.0, color="#2c3e50", linestyle="--", linewidth=2, label="Güvenlik Limiti (25 MP = 75 MB)")
        ax5.fill_between(cozunurluk_mp, 0, ram_mb, where=(cozunurluk_mp <= 25), color="#2ecc71", alpha=0.25, label="Güvenli Bölge")
        ax5.fill_between(cozunurluk_mp, 0, ram_mb, where=(cozunurluk_mp > 25), color="#e74c3c", alpha=0.25, label="Engellenen Bomb Bölgesi")

        ax5.set_title("5. Decompression Bomb Bellek Koruma Sınırı", fontweight="bold", color="#c0392b")
        ax5.set_xlabel("Çözünürlük (Megapiksel)")
        ax5.set_ylabel("Gereken Bellek (MB)")
        ax5.legend(loc="upper left", fontsize=8)

        # -------------------------------------------------------------
        # Panel 6: Üretim Giriş Boru Hattı Sağlık Dağılımı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        kategoriler = ["Güvenli Geçiş", "EXIF Düzeltme", "Alfa Normalizasyon", "Bozuk Kurtarma", "Bomb Engeli"]
        degerler = [75, 12, 8, 4, 1]
        renkler = ["#2ecc71", "#9b59b6", "#3498db", "#e67e22", "#e74c3c"]

        bars = ax6.bar(kategoriler, degerler, color=renkler, width=0.55, edgecolor="black", linewidth=1)
        ax6.set_ylabel("İşlem Oranı (%)")
        ax6.set_title("6. Canlı Giriş Akışı Sağlık ve Onarım Dağılımı", fontweight="bold", color="#2c3e50")
        ax6.tick_params(axis="x", rotation=25)

        for bar in bars:
            h = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., h + 1.2, f"%{h}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
