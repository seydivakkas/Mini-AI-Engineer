"""
6-Panelli Üretim Tensör Teftiş ve Anomali Teşhis Paneli (Tensor Inspector Dashboard).
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TensorDenetimGorsellestirici:
    """Tensör doğrulama telemetrisini ve anomalileri 6 panelli teşhis panosunda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        orijinal_tensor: np.ndarray,
        denetim_raporu: Dict[str, Any],
        temizlenmis_rapor: Optional[Dict[str, Any]] = None,
        hedef_path: str = "ciktilar/tensor_denetim_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.90)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 42: Üretim Girdi Tensörleri Doğrulama ve Anomali Teftiş Paneli (AI Batch Inspector)",
            fontsize=15, fontweight="bold", y=0.98
        )

        karar = denetim_raporu["karar"]
        stat = denetim_raporu["istatistikler"]

        # -------------------------------------------------------------
        # Panel 1: Denetim Karar Kartı & Telemetri
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        bg_c = "#2ecc71" if karar == "GECERLI" else "#f39c12" if karar == "DUZELTILEBILIR_UYARI" else "#e74c3c"
        kart_metni = (
            f"TENSÖR TEFTİŞ SONUCU: {karar}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Denetim Süresi     : {denetim_raporu['denetim_suresi_ms']:.3f} ms\n"
            f"• Güvenli Geçiş      : {'EVET (APPROVED)' if denetim_raporu['guvenli_gecis'] else 'HAYIR (REJECTED)'}\n"
            f"• Girdi Şekli (Shape): {denetim_raporu['sekil']}\n"
            f"• Veri Tipi (Dtype)  : {denetim_raporu['dtype']}\n"
            f"• Bellek Ayak İzi    : {denetim_raporu['bellek_mb']:.3f} MB\n"
            f"• C-Contiguous       : {'EVET' if denetim_raporu['c_contiguous'] else 'HAYIR (DAĞINIK)'}\n"
            f"• Toplam İhlal Sayısı: {denetim_raporu['toplam_ihlal_sayisi']} Adet"
        )
        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor=bg_c, alpha=0.25, edgecolor=bg_c, linewidth=2),
            fontsize=9.5, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Tensör Doğrulama Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Şekil & Boyut Doğrulama Matrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        kategoriler = ["Batch (N)", "Kanal (C)", "Yükseklik (H)", "Genişlik (W)"]
        sekil = denetim_raporu["sekil"]
        while len(sekil) < 4:
            sekil.append(0)

        # Eğer NHWC ise görselleştirme için boyutları normalize et
        gosterim_sekli = sekil[:4]
        b_plot = ax2.bar(kategoriler, gosterim_sekli, color="#3498db", alpha=0.85, edgecolor="black")
        for rect in b_plot:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2.0, h + max(gosterim_sekli) * 0.02, f"{int(h)}", ha="center", va="bottom", fontweight="bold")

        ax2.set_title("2. Tensör Şekli ve Boyut Analizi", fontweight="bold", color="#1f77b4")
        ax2.set_ylabel("Boyut Büyüklüğü (Piksel / Kanal)")

        # -------------------------------------------------------------
        # Panel 3: Sayısal Anomali Dağılımı (NaN/Inf/Aralık Dışı)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        anomali_turleri = ["NaN Sayısı", "Inf (Sonsuz)", "Aralık Dışı Değer"]
        anomali_degerleri = [
            stat.get("nan_sayisi", 0),
            stat.get("inf_sayisi", 0),
            stat.get("aralik_disi_piksel", 0)
        ]
        renkler_anomali = ["#e74c3c", "#c0392b", "#e67e22"]
        b_anomali = ax3.bar(anomali_turleri, anomali_degerleri, color=renkler_anomali, edgecolor="black")
        for rect in b_anomali:
            h = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width() / 2.0, h + max(max(anomali_degerleri), 1) * 0.02, f"{int(h)}", ha="center", va="bottom", fontweight="bold")

        ax3.set_title("3. Sayısal Kararsızlık & Anomali Dağılımı", fontweight="bold", color="#c0392b")
        ax3.set_ylabel("Hatalı Eleman Sayısı")

        # -------------------------------------------------------------
        # Panel 4: Tensör Değer Histogramı ve Aralık Sınırları
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        duzlesmis = orijinal_tensor[np.isfinite(orijinal_tensor)].flatten()
        if len(duzlesmis) > 0:
            sns.histplot(duzlesmis[:5000], bins=35, kde=True, ax=ax4, color="#9b59b6")
            ax4.axvline(-3.5, color="red", linestyle="--", label="Min Sınır (-3.5)")
            ax4.axvline(3.5, color="red", linestyle="--", label="Max Sınır (+3.5)")
            ax4.legend(loc="upper right", fontsize=8)
        ax4.set_title("4. Girdi Değer Dağılımı (Gözlemlenen Histogram)", fontweight="bold", color="#8e44ad")
        ax4.set_xlabel("Tensör Değerleri")

        # -------------------------------------------------------------
        # Panel 5: Bellek & Veri Düzeni Analizi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")

        ihlal_listesi_str = "TESPİT EDİLEN İHLALLER & AKSİYONLAR:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        if denetim_raporu["ihlaller"]:
            for ih in denetim_raporu["ihlaller"]:
                onem = "[KRİTİK]" if ih["kritik"] else "[UYARI]"
                ihlal_listesi_str += f"• {onem} {ih['kod']}:\n  {ih['mesaj']}\n\n"
        else:
            ihlal_listesi_str += "Tüm tensör kuralları eksiksiz karşılandı. İhlal yok."

        if temizlenmis_rapor:
            ihlal_listesi_str += "UYGULANAN TEMİZLEME İŞLEMLERİ:\n"
            for op in temizlenmis_rapor.get("yapilan_islemler", []):
                ihlal_listesi_str += f" -> {op}\n"

        ax5.text(
            0.05, 0.5, ihlal_listesi_str, transform=ax5.transAxes, va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdfefe", edgecolor="#7f8c8d", linewidth=1.5),
            fontsize=7.8, family="monospace"
        )
        ax5.set_title("5. İhlal Listesi & Temizleme Aksiyonları", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 6: Üretim Hattı Güvenilirlik & SLA Performans Radarı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        metrik_isimleri = ["Doğruluk", "Bellek Uyumu", "Sayısal Kararlılık", "SLA Hızı", "Şekil Uyumu"]
        
        # Skorlar (0 - 100)
        skorlar = [
            100 if karar == "GECERLI" else 75 if karar == "DUZELTILEBILIR_UYARI" else 20,
            100 if denetim_raporu["c_contiguous"] and denetim_raporu["bellek_mb"] < 128.0 else 50,
            100 if stat.get("nan_sayisi", 0) == 0 and stat.get("inf_sayisi", 0) == 0 else 0,
            100 if denetim_raporu["denetim_suresi_ms"] < 2.0 else 70,
            100 if not denetim_raporu.get("nhwc_tespit_edildi", False) else 60
        ]
        b_skor = ax6.barh(metrik_isimleri, skorlar, color="#1abc9c", alpha=0.85, edgecolor="black")
        for rect in b_skor:
            w = rect.get_width()
            ax6.text(w + 1.5, rect.get_y() + rect.get_height() / 2.0, f"%{int(w)}", va="center", fontweight="bold")

        ax6.set_xlim(0, 115)
        ax6.set_title("6. Üretim Hattı Güvenilirlik & Guardrail Skoru", fontweight="bold", color="#16a085")
        ax6.set_xlabel("Performans Puanı (%)")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
