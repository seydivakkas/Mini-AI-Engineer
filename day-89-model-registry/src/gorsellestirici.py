"""
Model Registry ve Yaşam Döngüsü Teşhis Panosu
---------------------------------------------
6 panelli yüksek çözünürlüklü Model Registry Mimarisi, Sürüm Yaşam Döngüsü Matrisi,
Kalite Kapısı Kriterleri, Üretim Modelleri Kıyası, Rollback Akışı ve SWOT Karar Matrisi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class RegistryGorsellestirici:
    """
    Model Registry durumlarını, aşama geçişlerini ve kalite kapısı kararlarını görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_registry_paneli(
        self,
        surumler: List[Dict[str, Any]],
        kalite_raporu_v2: Dict[str, Any],
        kalite_raporu_v3: Dict[str, Any],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı Model Registry ve Yaşam Döngüsü Panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 89: Model Kayıt Sistemi, Model Sürümleme ve Staging/Production Yaşam Döngüsü Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        # -------------------------------------------------------------
        # PANEL 1: Model Registry Mimarisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        kavram_metin = (
            "         MODEL REGISTRY & GOVERNANCE MİMARİSİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. IMMUTABLE VERSIONING (Değişmez Sürümleme):\n"
            "     • Her model eğitimi v1, v2, v3 olarak artan sürümle mühürlenir.\n"
            "     • Kaynak Run ID, Git Hash ve Tensör Şeması kaydedilir.\n\n"
            "  2. 4 AŞAMALI YAŞAM DÖNGÜSÜ (Lifecycle):\n"
            "     • NONE        : Ham kayıtlı model sürümü.\n"
            "     • STAGING     : Kalite Kapısı testindeki aday sürüm.\n"
            "     • PRODUCTION  : Canlı trafiği karşılayan aktif model.\n"
            "     • ARCHIVED    : Geri alma (rollback) için bekletilen eski sürüm.\n\n"
            "  3. AUTOMATED QUALITY GATES & ROLLBACK:\n"
            "     • Staging testlerini geçen model tek tıkla Production'a geçer.\n"
            "     • Beklenmeyen arızada anında sıfır kesintili Rollback yapılır."
        )
        ax1.text(
            0.5, 0.5, kavram_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Model Registry & Yaşam Döngüsü Akışı", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Model Sürümleri ve Aşamalar Matrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.axis("off")

        satirlar_metin = "  SÜRÜM    AŞAMA        DURUM               ACCURACY    GECİKME\n"
        satirlar_metin += "─────────────────────────────────────────────────────────────\n"
        for s in surumler:
            v_no = f"v{s['surum_no']}"
            asama = s["asama"]
            m = s["metrikler"]
            acc = f"%{m.get('val_acc', 0.0):.1f}"
            lat = f"{m.get('latency_ms', 0.0):.2f} ms"
            durum_aciklama = "Aktif Üretim" if asama == "PRODUCTION" else ("Arşivlendi" if asama == "ARCHIVED" else "Aday/Test")
            satirlar_metin += f"  {v_no:<8} {asama:<12} {durum_aciklama:<18} {acc:<11} {lat}\n"

        ax2.text(
            0.5, 0.5, satirlar_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f0fff4", edgecolor="#38a169", linewidth=1.8)
        )
        ax2.set_title("2. Model Sürümleri ve Yaşam Döngüsü Durum Matrisi", fontsize=12, fontweight="bold", color="#22543d")

        # -------------------------------------------------------------
        # PANEL 3: Kalite Kapısı (Quality Gate) Doğrulama Kriterleri
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        kriterler = ["Doğruluk (Acc %)", "Gecikme (ms)", "Kalibrasyon (ECE)"]
        v2_degerler = [
            kalite_raporu_v2["metrikler"]["val_acc"],
            kalite_raporu_v2["metrikler"]["latency_ms"],
            kalite_raporu_v2["metrikler"]["ece"] * 100
        ]
        v3_degerler = [
            kalite_raporu_v3["metrikler"]["val_acc"],
            kalite_raporu_v3["metrikler"]["latency_ms"],
            kalite_raporu_v3["metrikler"]["ece"] * 100
        ]

        x_k = np.arange(len(kriterler))
        w = 0.35

        ax3.bar(x_k - w/2, v2_degerler, width=w, color="#38a169", label=f"v2 (Kabul Edildi: {kalite_raporu_v2['gecti_mi']})")
        ax3.bar(x_k + w/2, v3_degerler, width=w, color="#e53e3e", label=f"v3 (Reddedildi: {kalite_raporu_v3['gecti_mi']})")

        ax3.set_title("3. Kalite Kapısı (Quality Gate) Karşılaştırması", fontsize=12, fontweight="bold", color="#c53030")
        ax3.set_xticks(x_k)
        ax3.set_xticklabels(kriterler, fontsize=9)
        ax3.legend(loc="upper right", frameon=True, fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 4: Sürümler Arası Doğruluk vs Gecikme Dağılımı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        v_isimler = [f"v{s['surum_no']}" for s in surumler]
        v_acclar = [s["metrikler"].get("val_acc", 0.0) for s in surumler]
        v_lat = [s["metrikler"].get("latency_ms", 0.0) for s in surumler]

        renkler = ["#3182ce" if s["asama"] == "PRODUCTION" else ("#718096" if s["asama"] == "ARCHIVED" else "#e53e3e") for s in surumler]

        ax4.scatter(v_lat, v_acclar, c=renkler, s=220, edgecolors="black", zorder=3)
        for idx, name in enumerate(v_isimler):
            asama_adi = surumler[idx]["asama"]
            ax4.annotate(
                f"{name} ({asama_adi})",
                (v_lat[idx], v_acclar[idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8.5,
                fontweight="bold"
            )

        ax4.set_title("4. Sürüm Doğruluk vs Gecikme Haritası", fontsize=12, fontweight="bold", color="#553c9a")
        ax4.set_xlabel("Çıkarım Gecikmesi (ms)", fontsize=10)
        ax4.set_ylabel("Doğruluk (%)", fontsize=10)
        ax4.set_ylim(0, 110)

        # -------------------------------------------------------------
        # PANEL 5: Rollback ve Sıfır Kesinti Mekanizması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        
        rollback_metin = (
            "         SIFIR KESİNTİLİ GERİ ALMA (ROLLBACK) AKIŞI\n"
            "─────────────────────────────────────────────────────────────\n"
            "  [ ADIM 1 ]: v1 Eğitildi ──> Quality Gate Geçti ──> PRODUCTION\n"
            "  [ ADIM 2 ]: v2 Eğitildi ──> Quality Gate Geçti ──> PRODUCTION\n"
            "              (v1 otomatik olarak ARCHIVED aşamasına alındı)\n\n"
            "  [ ADIM 3 ]: Canlıda v2 için beklenmeyen hata alarmı tetiklendi!\n"
            "  [ ADIM 4 ]: motor.geri_al(model_adi) çağrıldı:\n"
            "              • v2 ──> ARCHIVED\n"
            "              • v1 ──> PRODUCTION (Stabil duruma anında dönüş!)\n\n"
            "  ✓ Sonuç: Servis durdurulmadan < 10 ms içinde güvenli geri alma."
        )
        ax5.text(
            0.5, 0.5, rollback_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#fffaf0", edgecolor="#dd6b20", linewidth=1.8)
        )
        ax5.set_title("5. Sıfır Kesintili Geri Alma (Zero-Downtime Rollback)", fontsize=12, fontweight="bold", color="#c05621")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       MODEL REGISTRY & GOVERNANCE SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Hangi modelin canlıda olduğu tek merkezden (Single Source of Truth).\n"
            "  • Kalite kapısı ile hatalı modellerin prod'a geçişi %100 engellenir.\n"
            "  • Tek komutla sıfır kesintili acil geri alma (Instant Rollback).\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Eski modeller arşivlendiği için disk depolama yönetimi gerekir.\n"
            "  • Dağıtık sunucularda senkronizasyon için merkezi DB şarttır.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • CI/CD (GitHub Actions / GitLab CI) ile tam otomatik Staging deploy.\n"
            "  • Canary / Shadow deployment ile A/B trafik testleri.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Kalite kapısı eşikleri gevşek tutulursa hatalı modeller sızabilir."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Model Registry SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
