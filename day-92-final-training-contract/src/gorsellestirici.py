"""
Day 92: 6-Panelli Veri Sözleşmesi ve Hazır Bulunuşluk Teşhis Panosu
------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class VeriSozlesmesiGorsellestirici:
    """
    Veri sözleşmesi denetim sonuçlarını, sınır aşımlarını, sızıntı analizini
    ve nihai karar kapısını 6 panelli profesyonel bir kontrol panosu olarak çizer.
    """

    def __init__(self, cizim_boyutu: tuple = (18, 12), dpi: int = 300):
        self.cizim_boyutu = cizim_boyutu
        self.dpi = dpi

    def olustur_sozlesme_paneli(
        self,
        denetim_sonucu: any,
        sizinti_raporu: Optional[any],
        kapi_karari: any,
        ornek_tensörler: np.ndarray,
        kayit_yolu: str,
    ) -> None:
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, eksenler = plt.subplots(2, 3, figsize=self.cizim_boyutu, dpi=self.dpi)
        fig.suptitle(
            "Day 92: Egitim Oncesi Veri Sozlesmesi ve Hazir Bulunusluk (Readiness Gate) Panosu",
            fontsize=16,
            fontweight="bold",
            color="#111827",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: Şema ve Tensör Boyut Uyumluluk Kontrolü
        # -------------------------------------------------------------
        ax1 = eksenler[0, 0]
        kategoriler = ["Hacim (N)", "Kanal (C)", "Yükseklik (H)", "Genişlik (W)", "Tip (Dtype)"]
        durumlar = [
            1.0 if denetim_sonucu.toplam_ornek >= 50 else 0.0,
            1.0 if not any("GIRDI_SEKIL" in i.kural_adi for i in denetim_sonucu.ihlal_listesi) else 0.0,
            1.0 if not any("GIRDI_SEKIL" in i.kural_adi for i in denetim_sonucu.ihlal_listesi) else 0.0,
            1.0 if not any("GIRDI_SEKIL" in i.kural_adi for i in denetim_sonucu.ihlal_listesi) else 0.0,
            1.0 if not any("DTYPE" in i.kural_adi for i in denetim_sonucu.ihlal_listesi) else 0.5,
        ]
        renkler1 = ["#10b981" if d == 1.0 else "#ef4444" if d == 0.0 else "#f59e0b" for d in durumlar]
        ax1.barh(kategoriler, durumlar, color=renkler1, alpha=0.85, edgecolor="#374151")
        ax1.set_xlim(0, 1.2)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["İhlal", "Uygun"], fontsize=9)
        ax1.set_title("1. Şema ve Boyut Uyumluluk Matrisi", fontsize=11, fontweight="bold", color="#1f2937")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: Sayısal Sınırlar ve Değer Dağılımı
        # -------------------------------------------------------------
        ax2 = eksenler[0, 1]
        duz_degerler = ornek_tensörler.flatten()
        duz_degerler = duz_degerler[~np.isnan(duz_degerler) & ~np.isinf(duz_degerler)]
        if len(duz_degerler) > 0:
            ax2.hist(duz_degerler, bins=35, color="#3b82f6", alpha=0.6, edgecolor="#1d4ed8", density=True)
            ax2.axvline(denetim_sonucu.min_deger, color="#f59e0b", linestyle="--", label=f"Min: {denetim_sonucu.min_deger:.2f}")
            ax2.axvline(denetim_sonucu.maks_deger, color="#f59e0b", linestyle="--", label=f"Maks: {denetim_sonucu.maks_deger:.2f}")
            ax2.axvline(denetim_sonucu.ortalama_deger, color="#10b981", lw=2, label=f"Ortalama: {denetim_sonucu.ortalama_deger:.2f}")
        ax2.set_title("2. Değer Sınırları & İstatistiksel Aralık", fontsize=11, fontweight="bold", color="#1f2937")
        ax2.set_xlabel("Tensör Değerleri", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: NaN / Inf ve Bozulma Durumu
        # -------------------------------------------------------------
        ax3 = eksenler[0, 2]
        gecerli = denetim_sonucu.gecerli_ornek_sayisi
        bozuk = denetim_sonucu.nan_inf_sayisi
        ax3.pie(
            [max(1, gecerli), max(0, bozuk)],
            labels=["Geçerli Değerler", "NaN / Inf Bozuk"],
            colors=["#10b981", "#ef4444" if bozuk > 0 else "#e5e7eb"],
            autopct="%1.1f%%",
            startangle=140,
            explode=(0, 0.1) if bozuk > 0 else (0, 0),
        )
        ax3.set_title(f"3. Veri Bütünlüğü (NaN/Inf: {bozuk})", fontsize=11, fontweight="bold", color="#1f2937")

        # -------------------------------------------------------------
        # PANEL 4: Sınıf Dağılımı ve Dengesizlik Oranları
        # -------------------------------------------------------------
        ax4 = eksenler[1, 0]
        if denetim_sonucu.sinif_dagilimi:
            siniflar = list(denetim_sonucu.sinif_dagilimi.keys())
            sayimlar = list(denetim_sonucu.sinif_dagilimi.values())
            ax4.bar([str(s) for s in siniflar], sayimlar, color="#6366f1", alpha=0.8, edgecolor="#4338ca")
            ax4.axhline(np.mean(sayimlar), color="#f59e0b", linestyle=":", label=f"Ortalama: {np.mean(sayimlar):.1f}")
        ax4.set_title("4. Sınıf Dağılımı ve Frekans Profili", fontsize=11, fontweight="bold", color="#1f2937")
        ax4.set_xlabel("Sınıf İndeksi", fontsize=9)
        ax4.set_ylabel("Örnek Sayısı", fontsize=9)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: Train - Val Veri Sızıntısı Analizi (Contamination)
        # -------------------------------------------------------------
        ax5 = eksenler[1, 1]
        if sizinti_raporu:
            labels = ["Temiz Val Örnekleri", "Sızan (Train ile Ortak)"]
            val_temiz = sizinti_raporu.val_toplam - sizinti_raporu.kesisen_ornek_sayisi
            kesisen = sizinti_raporu.kesisen_ornek_sayisi
            renkler5 = ["#3b82f6", "#ef4444" if kesisen > 0 else "#e5e7eb"]
            ax5.bar(labels, [val_temiz, kesisen], color=renkler5, edgecolor="#374151", alpha=0.85)
            for i, v in enumerate([val_temiz, kesisen]):
                ax5.text(i, v + 1, str(v), ha="center", fontweight="bold", fontsize=10)
        ax5.set_title("5. Train-Val Veri Sızıntısı (Data Leakage)", fontsize=11, fontweight="bold", color="#1f2937")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: Hazır Bulunuşluk Kapısı Karar Kartı
        # -------------------------------------------------------------
        ax6 = eksenler[1, 2]
        ax6.axis("off")

        durum_renk = (
            "#10b981" if kapi_karari.egitim_baslatilabilir_mi and not kapi_karari.uyarilar
            else "#f59e0b" if kapi_karari.egitim_baslatilabilir_mi
            else "#ef4444"
        )

        metin_satirlari = [
            f"HAZIR BULUNUŞLUK KARARI:",
            f"» {kapi_karari.durum.value}",
            "─" * 40,
            f"• Toplam Örnek Sayısı : {denetim_sonucu.toplam_ornek}",
            f"• NaN / Inf Sayısı    : {denetim_sonucu.nan_inf_sayisi}",
            f"• Sızıntı Oranı (Val) : %{sizinti_raporu.sizinti_orani_val * 100:.2f}" if sizinti_raporu else "• Sızıntı: N/A",
            f"• Toplam İhlal Sayısı : {kapi_karari.toplam_ihlal_sayisi}",
            "─" * 40,
            f"• Eğitim Onayı        : {'VERİLDİ (BAŞLATILABİLİR)' if kapi_karari.egitim_baslatilabilir_mi else 'REDDEDİLDİ (BLOKE)'}",
        ]

        if kapi_karari.bloke_eden_hatalar:
            metin_satirlari.append("\nBLOKE EDEN HATALAR:")
            for h in kapi_karari.bloke_eden_hatalar[:2]:
                metin_satirlari.append(f" ! {h[:36]}...")

        kutu_metni = "\n".join(metin_satirlari)
        ax6.text(
            0.05,
            0.5,
            kutu_metni,
            fontsize=10,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8fafc", edgecolor=durum_renk, lw=2.5),
        )
        ax6.set_title("6. Hazır Bulunuşluk Karar Özeti", fontsize=11, fontweight="bold", color="#1f2937")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
