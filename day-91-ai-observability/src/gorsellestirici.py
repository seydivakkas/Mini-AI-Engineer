"""
Day 91: 6-Panelli AI Gözlemlenebilirlik ve Performans Teşhis Panosu
------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib stil ayarları
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class ObservabilityGorsellestirici:
    """
    Canlı AI metriklerini, gecikme dağılımlarını, PSI/KS drift analizlerini
    ve alarm durumlarını 6 panelli yüksek çözünürlüklü dashboard olarak görselleştirir.
    """

    def __init__(self, cizim_boyutu: tuple = (18, 12), dpi: int = 300):
        self.cizim_boyutu = cizim_boyutu
        self.dpi = dpi

    def olustur_gozlemlenebilirlik_paneli(
        self,
        zaman_serisi_verisi: Dict[str, np.ndarray],
        metrik_ozeti: Any,
        drift_raporu: Optional[Any],
        referans_ozellik: np.ndarray,
        canli_ozellik: np.ndarray,
        kayit_yolu: str,
    ) -> None:
        """6 Panelli profesyonel dashboard üretir ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, eksenler = plt.subplots(2, 3, figsize=self.cizim_boyutu, dpi=self.dpi)
        fig.suptitle(
            "Day 91: Canli AI Sistemlerinde Gozlemlenebilirlik (Observability & Drift Panosu)",
            fontsize=16,
            fontweight="bold",
            color="#111827",
            y=0.98,
        )

        gecikmeler = zaman_serisi_verisi.get("gecikmeler_ms", np.array([]))
        zamanlar = zaman_serisi_verisi.get("zaman_damgalari", np.array([]))

        # -------------------------------------------------------------
        # PANEL 1: Gerçek Zamanlı Trafik & Gecikme Zaman Serisi
        # -------------------------------------------------------------
        ax1 = eksenler[0, 0]
        if len(gecikmeler) > 0:
            goreli_zaman = zamanlar - zamanlar[0] if len(zamanlar) > 0 else np.arange(len(gecikmeler))
            ax1.plot(goreli_zaman, gecikmeler, color="#2563eb", alpha=0.7, lw=1.5, label="İstek Gecikmesi (ms)")
            ax1.axhline(metrik_ozeti.sla_gecikme_esigi_ms if hasattr(metrik_ozeti, 'sla_gecikme_esigi_ms') else 25.0,
                        color="#dc2626", linestyle="--", lw=1.8, label="SLA Eşiği (25 ms)")
            ax1.axhline(metrik_ozeti.p99_gecikme_ms, color="#f59e0b", linestyle=":", lw=1.5, label=f"P99: {metrik_ozeti.p99_gecikme_ms:.1f} ms")
        ax1.set_title("1. Gerçek Zamanlı İstek Gecikmesi & SLA", fontsize=11, fontweight="bold", color="#1f2937")
        ax1.set_xlabel("Süre (saniye)", fontsize=9)
        ax1.set_ylabel("Gecikme (ms)", fontsize=9)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: Gecikme Dağılımı ve Yüzdelikler (P50, P95, P99)
        # -------------------------------------------------------------
        ax2 = eksenler[0, 1]
        if len(gecikmeler) > 0:
            ax2.hist(gecikmeler, bins=30, color="#3b82f6", alpha=0.6, edgecolor="#1d4ed8", density=True)
            ax2.axvline(metrik_ozeti.p50_gecikme_ms, color="#10b981", lw=2, linestyle="-", label=f"P50: {metrik_ozeti.p50_gecikme_ms:.1f} ms")
            ax2.axvline(metrik_ozeti.p95_gecikme_ms, color="#f59e0b", lw=2, linestyle="--", label=f"P95: {metrik_ozeti.p95_gecikme_ms:.1f} ms")
            ax2.axvline(metrik_ozeti.p99_gecikme_ms, color="#ef4444", lw=2, linestyle=":", label=f"P99: {metrik_ozeti.p99_gecikme_ms:.1f} ms")
        ax2.set_title("2. Gecikme Dağılım Histogramı & SLA", fontsize=11, fontweight="bold", color="#1f2937")
        ax2.set_xlabel("Gecikme (ms)", fontsize=9)
        ax2.set_ylabel("Olasılık Yoğunluğu", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 3: Baseline vs Canlı Özellik Dağılımı (KDE Shift)
        # -------------------------------------------------------------
        ax3 = eksenler[0, 2]
        if len(referans_ozellik) > 0 and len(canli_ozellik) > 0:
            ref_tekil = referans_ozellik[:, 0] if referans_ozellik.ndim > 1 else referans_ozellik
            canli_tekil = canli_ozellik[:, 0] if canli_ozellik.ndim > 1 else canli_ozellik

            ax3.hist(ref_tekil, bins=25, alpha=0.5, color="#10b981", density=True, label="Baseline (Referans)", edgecolor="#047857")
            ax3.hist(canli_tekil, bins=25, alpha=0.5, color="#ef4444", density=True, label="Production (Canlı)", edgecolor="#b91c1c")
        ax3.set_title("3. Özellik Dağılımı Kayması (Feature Drift)", fontsize=11, fontweight="bold", color="#1f2937")
        ax3.set_xlabel("Öznitelik Değeri (Embedding Boyut 0)", fontsize=9)
        ax3.set_ylabel("Yoğunluk", fontsize=9)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 4: Population Stability Index (PSI) Karşılaştırması
        # -------------------------------------------------------------
        ax4 = eksenler[1, 0]
        if drift_raporu and drift_raporu.oznitelik_detaylari:
            isimler = list(drift_raporu.oznitelik_detaylari.keys())[:8]
            psi_degerleri = [drift_raporu.oznitelik_detaylari[k].psi_degeri for k in isimler]
            renkler = ["#ef4444" if v >= 0.20 else "#f59e0b" if v >= 0.10 else "#10b981" for v in psi_degerleri]

            cubuklar = ax4.barh(isimler, psi_degerleri, color=renkler, edgecolor="#374151", alpha=0.85)
            ax4.axvline(0.10, color="#f59e0b", linestyle="--", lw=1.5, label="Uyarı Eşiği (0.10)")
            ax4.axvline(0.20, color="#ef4444", linestyle="--", lw=1.5, label="Kritik Eşik (0.20)")
            for c, val in zip(cubuklar, psi_degerleri):
                ax4.text(val + 0.005, c.get_y() + c.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
        ax4.set_title("4. Population Stability Index (PSI) Profili", fontsize=11, fontweight="bold", color="#1f2937")
        ax4.set_xlabel("PSI Değeri", fontsize=9)
        ax4.legend(loc="lower right", fontsize=8)
        ax4.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 5: KS-Test İstatistiği ve P-Değeri Analizi
        # -------------------------------------------------------------
        ax5 = eksenler[1, 1]
        if drift_raporu and drift_raporu.oznitelik_detaylari:
            isimler = list(drift_raporu.oznitelik_detaylari.keys())[:8]
            ks_ist = [drift_raporu.oznitelik_detaylari[k].ks_istatistigi for k in isimler]
            ks_p = [drift_raporu.oznitelik_detaylari[k].ks_p_degeri for k in isimler]

            x_eksen = np.arange(len(isimler))
            genislik = 0.35

            ax5.bar(x_eksen - genislik / 2, ks_ist, genislik, label="KS Mesafesi (D)", color="#6366f1", alpha=0.8)
            ax5.bar(x_eksen + genislik / 2, ks_p, genislik, label="p-değeri", color="#ec4899", alpha=0.8)
            ax5.axhline(0.05, color="#dc2626", linestyle=":", lw=1.5, label="Alfa Eşiği (0.05)")
            ax5.set_xticks(x_eksen)
            ax5.set_xticklabels(isimler, rotation=30, fontsize=8)
        ax5.set_title("5. Kolmogorov-Smirnov (KS-Test) İstatistikleri", fontsize=11, fontweight="bold", color="#1f2937")
        ax5.set_ylabel("Değer", fontsize=9)
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 6: Sistem Sağlık Durumu & Alarm Özet Panosu
        # -------------------------------------------------------------
        ax6 = eksenler[1, 2]
        ax6.axis("off")

        durum_renk = "#10b981" if not (drift_raporu and drift_raporu.genel_durum == "KRITIK_KAYMA") else "#ef4444"
        durum_metni = "SAĞLIKLI" if durum_renk == "#10b981" else "KRİTİK DRIFT TESPİT EDİLDİ"

        metin_satirlari = [
            f"SİSTEM SAĞLIK DURUMU: {durum_metni}",
            "─" * 42,
            f"• Toplam İşlenen İstek : {metrik_ozeti.toplam_istek}",
            f"• Anlık İşlem Hacmi (RPS): {metrik_ozeti.anlik_rps:.1f} req/s",
            f"• Ortalama Gecikme     : {metrik_ozeti.ortalama_gecikme_ms:.2f} ms",
            f"• P50 / P95 / P99       : {metrik_ozeti.p50_gecikme_ms:.1f} / {metrik_ozeti.p95_gecikme_ms:.1f} / {metrik_ozeti.p99_gecikme_ms:.1f} ms",
            f"• SLA İhlal Oranı       : %{metrik_ozeti.sla_ihlal_orani * 100:.1f}",
            f"• Hata Oranı            : %{metrik_ozeti.hata_orani * 100:.2f}",
            "─" * 42,
            f"• Kayan Öznitelik Oranı : %{drift_raporu.sistem_drift_orani * 100:.1f}" if drift_raporu else "• Drift Bilgisi: N/A",
            f"• Tahmin (Output) Drift : {'EVET (Alarm)' if drift_raporu and drift_raporu.tahmin_kaymasi_var_mi else 'HAYIR (Kararlı)'}",
            f"• Tahmin PSI Değeri     : {drift_raporu.tahmin_psi:.4f}" if drift_raporu else "",
        ]

        kutu_metni = "\n".join(metin_satirlari)
        ax6.text(
            0.05,
            0.5,
            kutu_metni,
            fontsize=10,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8fafc", edgecolor=durum_renk, lw=2.0),
        )
        ax6.set_title("6. Sistem Sağlık & Gözlemlenebilirlik Özeti", fontsize=11, fontweight="bold", color="#1f2937")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
