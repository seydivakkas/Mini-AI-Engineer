"""
6-Panelli Tabüler Veri Kalitesi ve Şema Teşhis Panosu (Data Quality Dashboard).
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class VeriKaliteGorsellestirici:
    """Veri kalitesi metriklerini, ihlalleri ve temizleme etkilerini 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        ham_df: pd.DataFrame,
        temiz_df: pd.DataFrame,
        denetim_raporu: Dict[str, Any],
        temizleme_raporu: Optional[Dict[str, Any]] = None,
        hedef_path: str = "ciktilar/veri_kalite_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 44: Pandas ile Üretim Seviyesi Şema Doğrulama ve Veri Kalitesi Teşhis Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        karar = denetim_raporu["karar"]
        skor = denetim_raporu["kalite_skoru"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Kalite Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        bg_c = "#2ecc71" if karar == "GECERLI_MUKEMMEL" else "#f39c12" if karar == "DUZELTILEBILIR_KIRLI_VERI" else "#e74c3c"
        kart_metni = (
            f"TABLO KALİTE KARARI: {karar}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Tablo Adı          : {denetim_raporu['tablo_adi']}\n"
            f"• Genel Kalite Skoru : %{skor:.1f} / 100\n"
            f"• Denetim Durumu     : {denetim_raporu['durum']}\n"
            f"• Toplam İhlal Sayısı: {denetim_raporu['toplam_ihlal_sayisi']} Adet (Kritik: {denetim_raporu['kritik_ihlal_sayisi']})\n"
            f"• Satır x Sütun      : {denetim_raporu['satir_sayisi']} x {denetim_raporu['sutun_sayisi']}\n"
            f"• Mükerrer Satır     : {denetim_raporu['cift_satir_sayisi']} Adet\n"
            f"• Teftiş Süresi      : {denetim_raporu['denetim_suresi_ms']:.2f} ms"
        )
        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=bg_c, alpha=0.25, edgecolor=bg_c, linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Yönetici Veri Kalite Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Kolon Bazlı Kalite & Doluluk Oranları
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kolon_raporlari = denetim_raporu.get("kolon_raporlari", {})
        kolonlar = list(kolon_raporlari.keys())
        kalite_oranlari = [kolon_raporlari[k]["kalite_orani"] for k in kolonlar]

        b_kolon = ax2.barh(kolonlar, kalite_oranlari, color="#3498db", alpha=0.85, edgecolor="black")
        for rect in b_kolon:
            w = rect.get_width()
            ax2.text(w + 1.0, rect.get_y() + rect.get_height() / 2.0, f"%{w:.1f}", va="center", fontsize=8, fontweight="bold")

        ax2.set_xlim(0, 115)
        ax2.set_title("2. Kolon Bazlı Veri Hijyen Oranları", fontweight="bold", color="#1f77b4")
        ax2.set_xlabel("Kalite Skoru (%)")

        # -------------------------------------------------------------
        # Panel 3: Hata ve İhlal Türü Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ihlal_sayaclari = {}
        for ih in denetim_raporu.get("ihlaller", []):
            k = ih["kod"]
            ihlal_sayaclari[k] = ihlal_sayaclari.get(k, 0) + 1

        if not ihlal_sayaclari:
            ihlal_sayaclari["KUSURSUZ"] = 0

        turler = list(ihlal_sayaclari.keys())
        sayilar = list(ihlal_sayaclari.values())
        b_ih = ax3.bar(turler, sayilar, color="#e74c3c", edgecolor="black", alpha=0.85)
        for rect in b_ih:
            h = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width() / 2.0, h + 0.1, f"{int(h)}", ha="center", va="bottom", fontweight="bold")

        ax3.set_title("3. Tespit Edilen İhlal Türleri", fontweight="bold", color="#c0392b")
        ax3.tick_params(axis="x", rotation=25)
        ax3.set_ylabel("İhlal Sayısı")

        # -------------------------------------------------------------
        # Panel 4: Temizleme Öncesi ve Sonrası Dağılım (Boxplot Karşılaştırması)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        # İlk sayısal sütunu bul
        sayisal_kolon = None
        for col in ham_df.select_dtypes(include=[np.number]).columns:
            if col in temiz_df.columns:
                sayisal_kolon = col
                break

        if sayisal_kolon:
            ham_dolu = ham_df[sayisal_kolon].dropna()
            temiz_dolu = temiz_df[sayisal_kolon].dropna()
            data_to_plot = [ham_dolu, temiz_dolu]
            bp = ax4.boxplot(data_to_plot, patch_artist=True, tick_labels=["Ham (Kirli)", "Temizlenmiş"])
            bp['boxes'][0].set_facecolor('#e74c3c')
            bp['boxes'][1].set_facecolor('#2ecc71')
            ax4.set_title(f"4. Sınır Kırpma & İmpütasyon Etkisi ({sayisal_kolon})", fontweight="bold", color="#8e44ad")
            ax4.set_ylabel("Değer Aralığı")
        else:
            ax4.text(0.5, 0.5, "Sayısal Kolon Bulunamadı", ha="center", va="center")
            ax4.axis("off")

        # -------------------------------------------------------------
        # Panel 5: Uygulanan Veri Hijyeni & Temizleme Günlüğü
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")

        gunluk_metni = "UYGULANAN TEMİZLEME VE İMPÜTASYON İŞLEMLERİ:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        if temizleme_raporu and temizleme_raporu.get("yapilan_islemler"):
            gunluk_metni += f"• Başlangıç Satır : {temizleme_raporu['baslangic_satir_sayisi']}\n"
            gunluk_metni += f"• Temiz Satır     : {temizleme_raporu['temizlenmis_satir_sayisi']}\n"
            gunluk_metni += f"• Çıkarılan Satır : {temizleme_raporu['silinen_satir_sayisi']}\n\n"
            for op in temizleme_raporu["yapilan_islemler"][:5]:
                gunluk_metni += f" -> {op}\n"
        else:
            gunluk_metni += "Veri zaten kusursuzdu, herhangi bir temizleme gerekmedi."

        ax5.text(
            0.05, 0.5, gunluk_metni, transform=ax5.transAxes, va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdfefe", edgecolor="#7f8c8d", linewidth=1.5),
            fontsize=8.0, family="monospace"
        )
        ax5.set_title("5. Temizleme ve İmpütasyon Günlüğü", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 6: Üretim Hattı Hazırlık & Güvenilirlik Radarı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        metrikler = ["Doluluk", "Tip Uyumu", "Aralık Güveni", "Tekillik", "Şema Uyumu"]
        skorlar = [
            float(np.mean([100.0 - kr["null_orani"] for kr in kolon_raporlari.values()])) if kolon_raporlari else 100,
            100.0 if not any(ih["kod"] == "GECERSIZ_DTYPE" for ih in denetim_raporu.get("ihlaller", [])) else 60,
            100.0 if not any(ih["kod"] == "ARALIK_DISI_DEGER" for ih in denetim_raporu.get("ihlaller", [])) else 50,
            100.0 if denetim_raporu.get("cift_satir_sayisi", 0) == 0 else 70,
            100.0 if denetim_raporu.get("kritik_ihlal_sayisi", 0) == 0 else 30
        ]

        b_hazirlik = ax6.barh(metrikler, skorlar, color="#1abc9c", alpha=0.85, edgecolor="black")
        for rect in b_hazirlik:
            w = rect.get_width()
            ax6.text(w + 1.0, rect.get_y() + rect.get_height() / 2.0, f"%{int(w)}", va="center", fontweight="bold")

        ax6.set_xlim(0, 115)
        ax6.set_title("6. Üretim Boru Hattı Hazırlık Skoru", fontweight="bold", color="#16a085")
        ax6.set_xlabel("Güvenilirlik Puanı (%)")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
