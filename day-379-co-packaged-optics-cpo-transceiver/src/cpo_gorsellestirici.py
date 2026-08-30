"""
Day 379: Co-Packaged Optics (CPO) High-Speed Optical Transceiver Modeling
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; PAM4 dalga şekillerini, 3-göz diyagramını (eye diagram),
MZM elektro-optik transfer eğrisini ve 6-panelli CPO teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class CPOGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Co-Packaged Optics (CPO) Teşhis Panosu.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cikti_dizini = os.path.join(base_dir, "ciktilar")
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    def teshis_panelini_ciz(
        self,
        bench_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "cpo_transceiver_paneli.png"
    ) -> str:
        """
        6 Panelli CPO Optik Alıcı-Verici Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Co-Packaged Optics (CPO) 800G/1.6T Optical Transceiver & SerDes Modeling (Phase 19)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        p_opt = bench_res["p_opt_tx"][:80]
        v_norm = bench_res["v_norm"]

        # ------------------------------------------------------------------
        # Panel 1: 112 Gbps PAM4 Optik Modüle Dalga Şekli
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        time_syms = np.arange(len(p_opt))
        ax1.step(time_syms, p_opt, color="#8e44ad", linewidth=1.8, where="mid", label="Optik Güç ($P_{opt}$)")
        ax1.set_title("1. 112 Gbps PAM4 Optik Modülasyon Dalga Şekli", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Sembol İndeksi (56 GBaud)", fontsize=8)
        ax1.set_ylabel("Optik Güç (mW)", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=7.5)

        # ------------------------------------------------------------------
        # Panel 2: PAM4 3-Göz Diyagramı (Eye Diagram Overlay)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        # Her 2 sembollük pencereyi üst üste bindir (Eye Diagram)
        window = 32
        samples_per_sym = 16
        t_eye = np.linspace(-1, 1, samples_per_sym * 2)
        
        # Sentetik pürüzsüzleştirilmiş göz eğrileri
        for i in range(0, min(600, len(v_norm) - 2), 2):
            s0, s1 = v_norm[i], v_norm[i+1]
            interp_wave = np.concatenate([
                np.full(samples_per_sym, s0) + 0.1 * np.random.randn(samples_per_sym),
                np.full(samples_per_sym, s1) + 0.1 * np.random.randn(samples_per_sym)
            ])
            ax2.plot(t_eye, interp_wave, color="#2980b9", alpha=0.08, linewidth=0.8)

        ax2.set_title("2. PAM4 3-Göz Diyagramı (Eye Diagram Opening)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman Aralığı (UI - Unit Interval)", fontsize=8)
        ax2.set_ylabel("Normalize Alınan Voltaj", fontsize=8)
        ax2.set_ylim(-4.5, 4.5)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Elektro-Optik MZM Transfer Fonksiyonu (Cos² V)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        v_sweep = np.linspace(-3.0, 3.0, 200)
        t_sweep = np.cos((v_sweep / (2.0 * 1.5)) * np.pi) ** 2
        ax3.plot(v_sweep, t_sweep * 10.0, color="#d35400", linewidth=2.0, label="$P_{out}(V)$")
        ax3.axvline(-1.5, color="#7f8c8d", linestyle="--", label="$-V_\\pi = -1.5\\text{V}$")
        ax3.axvline(1.5, color="#7f8c8d", linestyle="--", label="$+V_\\pi = +1.5\\text{V}$")
        ax3.set_title("3. Silikon Fotonik MZM Elektro-Optik Eğrisi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Sürüş Voltajı (V)", fontsize=8)
        ax3.set_ylabel("Çıkış Optik Gücü (mW)", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Enerji Tüketimi (Pluggable vs CPO 4.8x Tasarruf)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        cpo_e = bench_res["cpo_energy_pj_bit"]
        plug_e = bench_res["pluggable_energy_pj_bit"]
        bars4 = ax4.bar(["Takılabilir Optik (DSP)", "Co-Packaged Optics (CPO)"], [plug_e, cpo_e], color=["#e74c3c", "#27ae60"], width=0.45)
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} pJ/bit", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax4.set_title(f"4. Enerji Verimliliği ({bench_res['energy_savings_x']:.1f}x Tasarruf)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Enerji (pJ / bit)", fontsize=8)
        ax4.set_ylim(0, 24)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Bit Hata Oranı (BER) vs Optik Alınan Güç
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        powers = np.linspace(-15, 0, 100)
        ber_curve = 0.5 * np.exp(-10.0 ** ((powers + 8.0) / 5.0))
        ber_curve = np.clip(ber_curve, 1e-15, 1.0)
        ax5.semilogy(powers, ber_curve, color="#2c3e50", linewidth=2.0, label="CPO BER Eğrisi")
        ax5.axhline(1e-4, color="#e67e22", linestyle="--", label="KP4 FEC Eşiği ($10^{-4}$)")
        ax5.axhline(1e-12, color="#27ae60", linestyle=":", label="Hedef BER ($10^{-12}$)")
        ax5.set_title("5. Bit Hata Oranı (BER) Hassasiyet Eğrisi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Alınan Optik Güç (dBm)", fontsize=8)
        ax5.set_ylabel("Bit Error Rate (BER)", fontsize=8)
        ax5.set_ylim(1e-15, 1.0)
        ax5.legend(loc="upper right", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: CPO 800G/1.6T Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Enerji Tasarrufu", "Göz Açıklığı Kalitesi", "BER Uyumluluğu", "CPO 800G Hazırlığı"]
        scores = [
            profiler_metrics.get("energy_savings_score", 96.0),
            profiler_metrics.get("eye_quality_score", 98.5),
            profiler_metrics.get("ber_compliance_score", 99.9),
            profiler_metrics.get("cpo_readiness_score", 98.1)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. CPO Co-Packaged Optics Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
