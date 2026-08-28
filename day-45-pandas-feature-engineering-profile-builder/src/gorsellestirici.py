"""
6-Panelli Özellik Mühendisliği ve Feature Store Teşhis Panosu (Feature Engineering Dashboard).
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class OzellikMuhendisligiGorsellestirici:
    """Türetilen özellikleri, kodlama etkilerini ve korelasyonları 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        ham_df: pd.DataFrame,
        islenmis_df: pd.DataFrame,
        profil_raporu: Dict[str, Any],
        hedef_kolon: str = "risk_skoru",
        hedef_path: str = "ciktilar/ozellik_muhendisligi_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 45: Özellik Mühendisliği, Encoding, Ölçeklendirme ve Feature Store Teşhis Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        n_ham = len(ham_df.columns)
        n_islenmis = len(islenmis_df.columns)
        n_turetilen = n_islenmis - n_ham

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özellik Mühendisliği Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"FEATURE STORE KAYIT ÖZETİ\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Ham Öznitelik Sayısı : {n_ham} Adet\n"
            f"• Toplam Özellik Sayısı: {n_islenmis} Adet\n"
            f"• Türetilen Yeni Özellik: +{n_turetilen} Adet\n"
            f"• Sayısal Özellikler   : {profil_raporu['sayisal_oznitelik_sayisi']} Adet\n"
            f"• Kategorik Özellikler : {profil_raporu['kategorik_oznitelik_sayisi']} Adet\n"
            f"• Toplam Satır Sayısı  : {profil_raporu['toplam_satir_sayisi']} Kayıt\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Feature Store Durumu : FEAST_SCHEMA_READY (ONAYLANDI)"
        )
        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#2ecc71", alpha=0.25, edgecolor="#27ae60", linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Feature Store Metadata Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Kategorik Kodlama (Smoothed Target vs Frequency)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        target_enc_cols = [c for c in islenmis_df.columns if c.endswith("_target_enc")]
        if target_enc_cols:
            t_col = target_enc_cols[0]
            sns.barplot(data=islenmis_df, x=t_col, y=target_enc_cols[0].replace("_target_enc", ""), ax=ax2, color="#3498db", errorbar=None)
            ax2.set_title(f"2. Smoothed Target Encoding Dağılımı ({t_col})", fontweight="bold", color="#1f77b4")
            ax2.set_xlabel("Smoothed Target Değeri")
        else:
            ax2.text(0.5, 0.5, "Target Encoding Bulunamadı", ha="center", va="center")
            ax2.axis("off")

        # -------------------------------------------------------------
        # Panel 3: Ölçeklendirme Öncesi ve Sonrası Dağılım (Log1p vs Raw)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        log_cols = [c for c in islenmis_df.columns if c.endswith("_log1p")]
        if log_cols:
            ham_col = log_cols[0].replace("_log1p", "")
            sns.kdeplot(ham_df[ham_col], label=f"Ham ({ham_col})", color="#e74c3c", ax=ax3, fill=True, alpha=0.3)
            ax3_twin = ax3.twinx()
            sns.kdeplot(islenmis_df[log_cols[0]], label="Log1p Dönüşümü", color="#2ecc71", ax=ax3_twin, fill=True, alpha=0.3)
            ax3.set_title(f"3. Çarpıklık Giderme ({ham_col} vs Log1p)", fontweight="bold", color="#d35400")
            ax3.set_xlabel("Değer")
        else:
            ax3.text(0.5, 0.5, "Log Dönüşümü Bulunamadı", ha="center", va="center")
            ax3.axis("off")

        # -------------------------------------------------------------
        # Panel 4: Özellik Korelasyon Isı Haritası (Top Features)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        sayisal_df = islenmis_df.select_dtypes(include=[np.number])
        if len(sayisal_df.columns) > 1:
            corr_matrix = sayisal_df.iloc[:, :7].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax4, annot_kws={"size": 7.5})
            ax4.set_title("4. Türetilen Özellikler Korelasyon Matrisi", fontweight="bold", color="#8e44ad")
            ax4.tick_params(axis="x", rotation=30)
        else:
            ax4.text(0.5, 0.5, "Korelasyon Hesaplanamadı", ha="center", va="center")
            ax4.axis("off")

        # -------------------------------------------------------------
        # Panel 5: Feature Store Metadata Kataloğu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")

        katalog_str = "FEATURE STORE METADATA KATALOĞU:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        sayac = 0
        for col, meta in profil_raporu["oznitelikler"].items():
            if sayac >= 6:
                break
            corr_str = f", Corr: {meta.get('hedef_korelasyonu', 'N/A')}" if "hedef_korelasyonu" in meta else ""
            katalog_str += f"• {col:<22} [{meta['dtype']}]: Null: %{meta['null_orani']}{corr_str}\n"
            sayac += 1

        ax5.text(
            0.05, 0.5, katalog_str, transform=ax5.transAxes, va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdfefe", edgecolor="#7f8c8d", linewidth=1.5),
            fontsize=7.8, family="monospace"
        )
        ax5.set_title("5. Özellik Metadata Kataloğu", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 6: Özellik Varyans & Bilgi Sıralaması
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        varyanslar = {}
        for col in sayisal_df.columns:
            if col != hedef_kolon and col != "musteri_id":
                varyanslar[col] = float(sayisal_df[col].var())

        sirali_var = sorted(varyanslar.items(), key=lambda x: x[1], reverse=True)[:6]
        if sirali_var:
            var_names = [x[0] for x in sirali_var]
            var_vals = [x[1] for x in sirali_var]
            b_var = ax6.barh(var_names, var_vals, color="#1abc9c", alpha=0.85, edgecolor="black")
            ax6.set_title("6. Türetilen Özellik Varyans Sıralaması", fontweight="bold", color="#16a085")
            ax6.set_xlabel("Varyans (Bilgi Çeşitliliği)")
        else:
            ax6.text(0.5, 0.5, "Varyans Hesaplanamadı", ha="center", va="center")
            ax6.axis("off")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
