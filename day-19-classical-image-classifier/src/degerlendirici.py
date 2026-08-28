"""Sınıflandırma Değerlendirici ve Görselleştirici Modülü.

Karmaşıklık Matrisi (Confusion Matrix), Model Performans Kıyaslaması ve
Öznitelik Önem Analizini tek bir yüksek çözünürlüklü teşhis panelinde sunar.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .siniflandirici import ModelSonucu


class SiniflandirmaDegerlendirici:
    """Sınıflandırıcı sonuçlarını görselleştiren ve raporlayan sınıf."""

    @classmethod
    def kapsamli_rapor_olustur(
        cls,
        sonuclar: List[ModelSonucu],
        sinif_isimleri: List[str],
        hedef_dosya: Optional[Path] = None,
    ) -> Path:
        """Modellerin sonuçlarını 4 panelli yüksek çözünürlüklü grafik çizelgesi olarak kaydeder.

        Paneller:
        1. Sol Üst: En İyi Modelin Karmaşıklık Matrisi (Confusion Matrix Heatmap)
        2. Sağ Üst: Modeller Arası Metrik Kıyaslaması (Accuracy, F1, Precision, Recall)
        3. Sol Alt: Random Forest Öznitelik Önem Dağılımı (HOG, LBP, Renk)
        4. Sağ Alt: Eğitim ve Çıkarım Gecikmesi (Latency / Speed Benchmark)

        Args:
            sonuclar: ModelSonucu nesneleri listesi.
            sinif_isimleri: Kategori adları listesi.
            hedef_dosya: Kaydedilecek PNG dosya yolu.

        Returns:
            Oluşturulan dosyanın yolu.
        """
        if hedef_dosya is None:
            hedef_dosya = Path("ciktilar/siniflandirma_raporu.png")
        hedef_dosya.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.patch.set_facecolor("#fafafa")

        # En yüksek F1 skorlu modeli belirle
        en_iyi_model = max(sonuclar, key=lambda s: s.f1_macro)

        # ----------------------------------------------------
        # Panel 1: Karmaşıklık Matrisi (Confusion Matrix)
        # ----------------------------------------------------
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor("#ffffff")
        cm = confusion_matrix(en_iyi_model.y_true, en_iyi_model.y_pred)
        im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ticks = np.arange(len(sinif_isimleri))
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.set_xticklabels(sinif_isimleri, rotation=30, ha="right", fontsize=10)
        ax1.set_yticklabels(sinif_isimleri, fontsize=10)

        # Hücre içi sayıları yazdır
        esik = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                renk = "white" if cm[i, j] > esik else "black"
                ax1.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color=renk, fontsize=12, fontweight="bold"
                )

        ax1.set_title(
            f"Karmaşıklık Matrisi (Confusion Matrix) — {en_iyi_model.model_adi}\n"
            f"(Genel Doğruluk: %{en_iyi_model.accuracy * 100:.1f})",
            fontsize=12, fontweight="bold", pad=10
        )
        ax1.set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
        ax1.set_ylabel("Gerçek Sınıf", fontsize=10)

        # ----------------------------------------------------
        # Panel 2: Model Metrik Kıyaslaması (Bar Chart)
        # ----------------------------------------------------
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor("#ffffff")

        model_adlari = [s.model_adi for s in sonuclar]
        x = np.arange(len(model_adlari))
        genislik = 0.20

        accs = [s.accuracy for s in sonuclar]
        f1s = [s.f1_macro for s in sonuclar]
        precs = [s.precision_macro for s in sonuclar]
        recs = [s.recall_macro for s in sonuclar]

        ax2.bar(x - 1.5 * genislik, accs, genislik, label="Doğruluk (Acc)", color="#1976d2")
        ax2.bar(x - 0.5 * genislik, f1s, genislik, label="F1-Macro", color="#388e3c")
        ax2.bar(x + 0.5 * genislik, precs, genislik, label="Hassasiyet (Prec)", color="#f57c00")
        ax2.bar(x + 1.5 * genislik, recs, genislik, label="Duyarlılık (Recall)", color="#7b1fa2")

        ax2.set_ylabel("Skor (0.0 - 1.0)", fontsize=10)
        ax2.set_title("Modeller Arası Performans Kıyaslaması", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_adlari, fontsize=10)
        ax2.set_ylim([0.0, 1.15])
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.legend(loc="upper right", frameon=True, facecolor="#f5f5f5")

        # ----------------------------------------------------
        # Panel 3: Random Forest Öznitelik Önem Dağılımı
        # ----------------------------------------------------
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor("#ffffff")

        # RF sonucunu bul
        rf_sonuc = next((s for s in sonuclar if s.feature_importances is not None), None)
        if rf_sonuc and rf_sonuc.feature_importances is not None:
            imp = rf_sonuc.feature_importances
            # Öznitelik grupları: HOG (72), LBP (10), Renk Momentleri (12)
            n_top = min(15, len(imp))
            top_indeksler = np.argsort(imp)[::-1][:n_top]
            top_skorlar = imp[top_indeksler]

            y_pos = np.arange(n_top)
            ax3.barh(y_pos, top_skorlar, color="#00897b", edgecolor="black", alpha=0.85)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f"Öznitelik #{idx}" for idx in top_indeksler], fontsize=9)
            ax3.invert_yaxis()
            ax3.set_xlabel("Gini Önemi (Feature Importance)", fontsize=10)
            ax3.set_title(
                f"En Etkili {n_top} Görsel Öznitelik (Random Forest)\n"
                f"(Toplam {len(imp)} Boyut İçerisinden)",
                fontsize=12, fontweight="bold", pad=10
            )
            ax3.grid(True, linestyle="--", alpha=0.5, axis="x")
        else:
            ax3.text(0.5, 0.5, "Öznitelik Önemi Mevcut Değil", ha="center", va="center")

        # ----------------------------------------------------
        # Panel 4: Hız / Gecikme Benchmark'ı (Eğitim vs. Çıkarım)
        # ----------------------------------------------------
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor("#ffffff")

        egitim_sureleri = [s.egitim_suresi_ms for s in sonuclar]
        tahmin_sureleri = [s.tahmin_suresi_ms for s in sonuclar]

        x4 = np.arange(len(model_adlari))
        genislik4 = 0.35

        bar_egitim = ax4.bar(x4 - genislik4 / 2, egitim_sureleri, genislik4, label="Eğitim Süresi (ms)", color="#d32f2f")
        bar_tahmin = ax4.bar(x4 + genislik4 / 2, tahmin_sureleri, genislik4, label="Çıkarım / Test Süresi (ms)", color="#0288d1")

        for b in list(bar_egitim) + list(bar_tahmin):
            yval = b.get_height()
            ax4.text(
                b.get_x() + b.get_width() / 2.0, yval + 0.5,
                f"{yval:.1f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        ax4.set_ylabel("Süre (milisaniye - ms)", fontsize=10)
        ax4.set_title("Hesaplama Maliyeti & Gecikme Analizi", fontsize=12, fontweight="bold", pad=10)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(model_adlari, fontsize=10)
        ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax4.legend(loc="upper left", frameon=True, facecolor="#f5f5f5")

        plt.suptitle(
            "DAY 19: GELENEKSEL MAKİNE ÖĞRENMESİ İLE GÖRSEL SINIFLANDIRMA\n"
            f"Öznitelikler: HOG + LBP + Renk | En İyi Model: {en_iyi_model.model_adi} (F1={en_iyi_model.f1_macro:.4f})",
            fontsize=14, fontweight="bold", color="#212121", y=0.98
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(hedef_dosya, bbox_inches="tight", dpi=150)
        plt.close(fig)

        return hedef_dosya
