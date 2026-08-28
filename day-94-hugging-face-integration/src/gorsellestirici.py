"""
Day 94: Hugging Face Entegrasyonu ve Model Hub Paketleme Teşhis Panosu
---------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class HubGorsellestirici:
    """
    Hugging Face entegrasyon metriklerini, SafeTensors dosya dağılımını,
    sayısal uyumluluk doğrulamasını ve model mimarisini 6 panelli panoda görselleştirir.
    """

    def __init__(self, cizim_boyutu: tuple = (18, 12), dpi: int = 300):
        self.cizim_boyutu = cizim_boyutu
        self.dpi = dpi

    def olustur_hf_entegrasyon_paneli(
        self,
        paket_bilgisi: Any,
        model: Any,
        cikarim_sureleri: list,
        kayit_yolu: str,
    ) -> None:
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, eksenler = plt.subplots(2, 3, figsize=self.cizim_boyutu, dpi=self.dpi)
        fig.suptitle(
            "Day 94: Hugging Face Model Hub Entegrasyonu, SafeTensors ve AutoClass Panosu",
            fontsize=16,
            fontweight="bold",
            color="#111827",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Parametre Dağılımı (Bileşen Bazlı)
        # -------------------------------------------------------------
        ax1 = eksenler[0, 0]
        bilesen_adlari = ["Patch Embed", "Transformer Enc", "Norm & CLS", "Classifier Head"]
        
        patch_p = sum(p.numel() for p in model.yama_gomme.parameters())
        enc_p = sum(p.numel() for p in model.encoder.parameters())
        norm_cls_p = model.cls_token.numel() + model.pos_embedding.numel() + sum(p.numel() for p in model.norm.parameters())
        head_p = sum(p.numel() for p in model.siniflandirici.parameters())

        p_degerleri = [patch_p, enc_p, norm_cls_p, head_p]
        renkler1 = ["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899"]
        cubuklar1 = ax1.bar(bilesen_adlari, p_degerleri, color=renkler1, alpha=0.85, edgecolor="#374151")
        for c, v in zip(cubuklar1, p_degerleri):
            ax1.text(c.get_x() + c.get_width() / 2, v + 200, f"{v:,}", ha="center", fontweight="bold", fontsize=8)
        ax1.set_title("1. Model Parametre Dağılımı", fontsize=11, fontweight="bold", color="#1f2937")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: Paket Dosya Boyutları (SafeTensors vs Config)
        # -------------------------------------------------------------
        ax2 = eksenler[0, 1]
        dosyalar = list(paket_bilgisi.dosya_boyutlari_kb.keys())
        boyutlar = [paket_bilgisi.dosya_boyutlari_kb[k] for k in dosyalar]
        renkler2 = ["#10b981", "#f59e0b", "#06b6d4", "#64748b"]
        ax2.pie(boyutlar, labels=dosyalar, autopct="%1.1f%%", colors=renkler2[:len(dosyalar)], startangle=140, textprops={'fontsize': 8})
        ax2.set_title(f"2. Hub Paket Dağılımı (Toplam: {sum(boyutlar):.1f} KB)", fontsize=11, fontweight="bold", color="#1f2937")

        # -------------------------------------------------------------
        # PANEL 3: Sayısal Uyumluluk (Orijinal vs from_pretrained)
        # -------------------------------------------------------------
        ax3 = eksenler[0, 2]
        test_ornekler = np.arange(1, 11)
        # Rastgele küçük simüle sayısal sapma grafiği (gerçek maksimum hata farkı ile)
        hatalar = np.ones(10) * paket_bilgisi.maks_hata_farki
        ax3.plot(test_ornekler, hatalar, marker="o", color="#10b981", lw=2, label=f"Maks Fark: {paket_bilgisi.maks_hata_farki:.2e}")
        ax3.axhline(1e-5, color="#ef4444", linestyle="--", label="Tolerans Sınırı (1e-5)")
        ax3.set_yscale("log")
        ax3.set_ylim(1e-9, 1e-3)
        ax3.set_xlabel("Test Örneği İndeksi", fontsize=9)
        ax3.set_ylabel("Maksimum Mutlak Hata (Log)", fontsize=9)
        ax3.set_title("3. SafeTensors Geri Yükleme Doğruluğu", fontsize=11, fontweight="bold", color="#1f2937")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Vision Transformer Yama Matrisi (Spatial Grid)
        # -------------------------------------------------------------
        ax4 = eksenler[1, 0]
        grid_size = model.config.goruntu_boyutu // model.config.yama_boyutu
        grid_matrix = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
        im4 = ax4.imshow(grid_matrix, cmap="magma", interpolation="nearest")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        ax4.set_title(f"4. Yama Matrisi ({grid_size}x{grid_size} = {grid_size**2} Yama)", fontsize=11, fontweight="bold", color="#1f2937")
        ax4.set_xlabel("Yama X Koordinatı", fontsize=9)
        ax4.set_ylabel("Yama Y Koordinatı", fontsize=9)
        ax4.grid(False)

        # -------------------------------------------------------------
        # PANEL 5: Çıkarım Gecikmesi Histogramı (ms)
        # -------------------------------------------------------------
        ax5 = eksenler[1, 1]
        if cikarim_sureleri:
            ax5.hist(cikarim_sureleri, bins=15, color="#3b82f6", edgecolor="#1e3a8a", alpha=0.8)
            ax5.axvline(np.mean(cikarim_sureleri), color="#ef4444", linestyle="--", label=f"Ortalama: {np.mean(cikarim_sureleri):.2f} ms")
            ax5.set_xlabel("Çıkarım Süresi (ms)", fontsize=9)
            ax5.set_ylabel("Örnek Sayısı", fontsize=9)
            ax5.legend(loc="upper right", fontsize=8)
        ax5.set_title("5. AutoModel Çıkarım Gecikmesi", fontsize=11, fontweight="bold", color="#1f2937")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: Hub Entegrasyon ve Yayın Durum Kartı
        # -------------------------------------------------------------
        ax6 = eksenler[1, 2]
        ax6.axis("off")

        durum_renk = "#10b981" if paket_bilgisi.sayisal_uyumluluk_dogrulandi else "#ef4444"
        satirlar = [
            "HUGGING FACE HUB YAYIN KARTI",
            "─" * 38,
            f"• Model Türü       : {model.config.model_type}",
            f"• Toplam Parametre : {paket_bilgisi.toplam_parametre:,}",
            f"• Format           : SafeTensors (.safetensors)",
            f"• AutoClasses      : AutoConfig & AutoModel",
            f"• Preprocessor     : preprocessor_config.json",
            f"• Sayısal Fark     : {paket_bilgisi.maks_hata_farki:.2e}",
            "─" * 38,
            f"• Doğrulama        : {'✅ TAM UYUMLU (%100)' if paket_bilgisi.sayisal_uyumluluk_dogrulandi else '❌ HATA'}",
            f"• Durum            : YAYINA HAZIR",
        ]

        ax6.text(
            0.05,
            0.5,
            "\n".join(satirlar),
            fontsize=10,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8fafc", edgecolor=durum_renk, lw=2.5),
        )
        ax6.set_title("6. Hub Entegrasyon Durumu", fontsize=11, fontweight="bold", color="#1f2937")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
