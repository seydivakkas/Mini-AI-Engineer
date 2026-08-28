"""
6-Panelli Streamlit SQLite AI Yönetim ve Telemetri Teşhis Panosu.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class DashboardGorsellestirici:
    """SQLite AI çıkarım logları, telemetri metrikleri ve insan denetimi verilerini 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        istatistikler: Dict[str, Any],
        df_loglar: pd.DataFrame,
        df_siniflar: pd.DataFrame,
        hedef_path: str = "ciktilar/streamlit_sqlite_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 65: SQLite Destekli CRUD, Model Çıkarım Logları ve Kalıcı AI Yönetim Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        n_req = istatistikler.get("toplam_istek", 0)
        ort_lat = istatistikler.get("ortalama_gecikme_ms", 0.0)
        ort_conf = istatistikler.get("ortalama_guven", 0.0)
        n_det = istatistikler.get("toplam_tespit", 0)
        n_dogru = istatistikler.get("dogrulanan_adet", 0)
        n_yanlis = istatistikler.get("yanlis_adet", 0)

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özeti Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"AI ÇIKARIM YÖNETİMİ & CRUD ÖZETİ\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Veritabanı Modu            : SQLite 3 (WAL Mode)\n"
            f"• Toplam Loglanan Çıkarım    : {n_req:,} İstek\n"
            f"• Toplam Tespit Edilen Nesne : {n_det:,} Adet\n"
            f"─────────────────────────────────────────────\n"
            f"• Ortalama Çıkarım Gecikmesi : {ort_lat:.2f} ms\n"
            f"• Ortalama Güven Skoru       : %{ort_conf*100:.1f}\n"
            f"• İnsan Denetim Kayıtları    : {n_dogru + n_yanlis} Adet ({n_dogru} Doğru / {n_yanlis} Hatalı)\n"
            f"─────────────────────────────────────────────\n"
            f"• Kalıcı Depolama Durumu     : %100 AKTİF & TUTARLI"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#2ecc71", alpha=0.18, edgecolor="#27ae60", linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. AI Yönetim Paneli Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Sınıf Frekans Dağılımı
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        if not df_siniflar.empty:
            sns.barplot(data=df_siniflar, x="sinif_adi", y="adet", hue="sinif_adi", palette="viridis", ax=ax2, edgecolor="#2c3e50", legend=False)
            for p in ax2.patches:
                val = int(p.get_height())
                ax2.annotate(f"{val}", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                             ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax2.set_xlabel("Tespit Edilen Sınıf")
            ax2.set_ylabel("Frekans / Adet")
            ax2.tick_params(axis='x', rotation=15)
        ax2.set_title("2. Nesne Sınıf Dağılımı", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 3: Çıkarım Güven Skoru Dağılımı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        if not df_loglar.empty:
            sns.histplot(df_loglar["ortalama_guven"], kde=True, color="#9b59b6", ax=ax3, bins=20)
            ax3.axvline(df_loglar["ortalama_guven"].mean(), color="#e74c3c", linestyle="--", linewidth=2, label="Ortalama Güven")
            ax3.set_xlabel("Ortalama Güven Skoru")
            ax3.set_ylabel("Frekans")
            ax3.legend(loc="upper left")
        ax3.set_title("3. Güven Skoru Dağılımı", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 4: Zaman Serisi Gecikme Profili & Kayan Ortalama
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        if not df_loglar.empty and len(df_loglar) > 5:
            gecikmeler = df_loglar["gecikme_ms"].values[::-1]  # Kronolojik sıra
            kayan_ort = pd.Series(gecikmeler).rolling(window=10, min_periods=1).mean()
            ax4.plot(gecikmeler, alpha=0.35, color="#3498db", label="Anlık Gecikme (ms)")
            ax4.plot(kayan_ort, linewidth=2.5, color="#e67e22", label="Kayan Ortalama (w=10)")
            ax4.set_xlabel("İstek Sırası (En Eski -> En Yeni)")
            ax4.set_ylabel("Gecikme (ms)")
            ax4.legend(loc="upper right")
        ax4.set_title("4. Çıkarım Gecikme Trendi", fontweight="bold", color="#d35400")

        # -------------------------------------------------------------
        # Panel 5: İnsan Denetimi (Human-in-the-Loop) Durumu
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        toplam_etiket = n_dogru + n_yanlis
        if toplam_etiket > 0:
            oranlar = [n_dogru, n_yanlis]
            etiketler = [f"Doğru ({n_dogru})", f"Hatalı ({n_yanlis})"]
            ax5.pie(oranlar, labels=etiketler, autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
                    startangle=90, explode=(0.05, 0), wedgeprops=dict(edgecolor="#2c3e50", linewidth=1.5))
        else:
            ax5.text(0.5, 0.5, "Henüz İnsan Denetim\nVerisi Yok", ha="center", va="center", fontsize=11, fontweight="bold")
        ax5.set_title("5. Human-in-the-Loop Denetim Oranı", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # Panel 6: SQLite + Streamlit SWOT Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_text = (
            "STREAMLIT & SQLITE AI YÖNETİM SWOT MATRİSİ\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "[S] GÜÇLÜ YÖNLER (Strengths):\n"
            " • Sıfır konfigürasyon, gömülü WAL ilişkisel DB\n"
            " • Hızlı interaktif arayüz ve veri görselleştirme\n\n"
            "[W] ZAYIF YÖNLER (Weaknesses):\n"
            " • SQLite tek yazıcı (single-writer) kilit limiti\n"
            " • Sayfa etkileşimlerinde tüm script yeniden koşar\n\n"
            "[O] FIRSATLAR (Opportunities):\n"
            " • Model sapması (drift) ve anomali tespiti\n"
            " • Uç cihazlarda (Edge) bağımsız denetim izi\n\n"
            "[T] TEHDİTLER (Threats):\n"
            " • Çok yüksek eşzamanlı yazmada kilitlenme"
        )
        ax6.text(
            0.5, 0.5, swot_text, transform=ax6.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f39c12", alpha=0.15, edgecolor="#e67e22", linewidth=2),
            fontsize=8.5, fontweight="bold", family="monospace"
        )
        ax6.set_title("6. Streamlit & SQLite Mimarisi SWOT Matrisi", fontweight="bold", color="#d35400")

        fig.subplots_adjust(top=0.93, bottom=0.10, left=0.10, right=0.95, hspace=0.36, wspace=0.32)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
