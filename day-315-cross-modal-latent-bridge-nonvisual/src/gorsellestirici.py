"""
Day 315: 6-Panel Diagnostic Dashboard Visualizer for Non-Visual Cross-Modal Latent Bridge.
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

from typing import Dict, Any, Optional
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .gorsel_olmayan_latent_kopru import CrossModalBenchmarkResult


class NonVisualCrossModalGorsellestirici:
    """
    Renders a 6-panel dark-mode diagnostic dashboard for Non-Visual Cross-Modal Latent Bridge.
    """
    
    @staticmethod
    def ciz(result: CrossModalBenchmarkResult, cikti_yolu: str = "ciktilar/gorsel_olmayan_kopru_paneli.png", 
            profil_ozeti: Optional[Dict[str, Any]] = None):
        """
        Generates and saves the 6-panel diagnostic dashboard.
        """
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)
        
        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        fig.patch.set_facecolor("#0b0f19")
        
        # -------------------------------------------------------------
        # Panel 1: Modality Zero-Shot Accuracies
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor("#111827")
        
        mods = ["Koku (E-Nose)\nKimyasal Dizi", "Termal IR\nRadyometrik", "Ultrasonik Sonar\nAkustik Doppler"]
        accs = [result.olfactory_zero_shot_acc_pct, result.thermal_zero_shot_acc_pct, result.sonar_zero_shot_acc_pct]
        colors1 = ["#10b981", "#f59e0b", "#38bdf8"]
        
        b1 = ax1.bar(mods, accs, color=colors1, width=0.45, edgecolor="#ffffff", lw=1.2, alpha=0.85)
        for bar, val in zip(b1, accs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"%{val:.1f}", 
                     ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")
                     
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Zero-Shot Doğruluk (%)", color="#94a3b8", fontsize=10)
        ax1.set_title("1. Görsel Olmayan Modaliteler Sıfır-Örnek Doğruluğu", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax1.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        
        # -------------------------------------------------------------
        # Panel 2: InfoNCE Loss Curve
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor("#111827")
        
        losses = result.training_loss_history
        epochs = np.arange(len(losses)) + 1
        ax2.plot(epochs, losses, color="#a855f7", lw=2.2, label="InfoNCE Köprü Kaybı")
        ax2.fill_between(epochs, 0, losses, color="#a855f7", alpha=0.15)
        
        ax2.set_xlabel("Eğitim Dönemi (Epoch)", color="#94a3b8", fontsize=10)
        ax2.set_ylabel("Karşıtsal Kayıp (InfoNCE Loss)", color="#94a3b8", fontsize=10)
        ax2.set_title("2. Çoklu Modalite Karşıtsal Hizalama Kaybı", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax2.grid(True, color="#1e293b", linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.3, facecolor="#1e293b", edgecolor="#334155", fontsize=9)
        
        # -------------------------------------------------------------
        # Panel 3: Olfactory Confusion Matrix
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.set_facecolor("#111827")
        
        cm_olf = result.modality_confusion_matrices["olfactory"]
        im3 = ax3.imshow(cm_olf, cmap="Blues", interpolation="nearest")
        ax3.set_title("3. Koku (E-Nose) Hata Matrisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax3.set_xlabel("Tahmin Edilen Sınıf", color="#94a3b8", fontsize=9)
        ax3.set_ylabel("Gerçek Sınıf", color="#94a3b8", fontsize=9)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # -------------------------------------------------------------
        # Panel 4: Thermal Confusion Matrix
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.set_facecolor("#111827")
        
        cm_thm = result.modality_confusion_matrices["thermal"]
        im4 = ax4.imshow(cm_thm, cmap="Oranges", interpolation="nearest")
        ax4.set_title("4. Termal Kızılötesi Hata Matrisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax4.set_xlabel("Tahmin Edilen Sınıf", color="#94a3b8", fontsize=9)
        ax4.set_ylabel("Gerçek Sınıf", color="#94a3b8", fontsize=9)
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # -------------------------------------------------------------
        # Panel 5: Sonar Confusion Matrix
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.set_facecolor("#111827")
        
        cm_snr = result.modality_confusion_matrices["sonar"]
        im5 = ax5.imshow(cm_snr, cmap="Greens", interpolation="nearest")
        ax5.set_title("5. Ultrasonik Sonar Hata Matrisi", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax5.set_xlabel("Tahmin Edilen Sınıf", color="#94a3b8", fontsize=9)
        ax5.set_ylabel("Gerçek Sınıf", color="#94a3b8", fontsize=9)
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # -------------------------------------------------------------
        # Panel 6: Telemetry Summary Box
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_facecolor("#111827")
        ax6.axis("off")
        
        p = profil_ozeti or {}
        
        kpi_text = (
            "DAY 315: NON-VISUAL CROSS-MODAL LATENT BRIDGE\n"
            "==============================================\n"
            f"• Genel Çapraz-Modalite Doğruluğu: %{p.get('overall_cross_modal_acc_pct', result.overall_cross_modal_acc_pct):.2f}\n"
            f"• Koku (E-Nose) Sıfır-Örnek Doğruluğu: %{p.get('olfactory_zero_shot_acc_pct', result.olfactory_zero_shot_acc_pct):.2f}\n"
            f"• Termal IR Sıfır-Örnek Doğruluğu: %{p.get('thermal_zero_shot_acc_pct', result.thermal_zero_shot_acc_pct):.2f}\n"
            f"• Sonar Sıfır-Örnek Doğruluğu: %{p.get('sonar_zero_shot_acc_pct', result.sonar_zero_shot_acc_pct):.2f}\n"
            f"• Çapraz-Modalite Kosinüs Hizalaması: {p.get('mean_cross_modal_alignment_cosine', result.mean_cross_modal_alignment_cosine):.4f}\n"
            f"• Gizil Uzay İzometri Skoru: {p.get('latent_isometry_score', result.latent_isometry_score):.4f}\n"
            f"• Entegrasyon Sınıfı: {p.get('integration_tier', 'SUPER_ALIGNED_CROSS_MODAL_SPACE')}\n"
            "==============================================\n"
            "DURUM: KİMYASAL, TERMAL, AKUSTİK VE METİNSEL\n"
            "        BİRLEŞİK GİZİL UZAY HİZALAMASI AKTİF"
        )
        
        ax6.text(
            0.05, 0.5, kpi_text,
            transform=ax6.transAxes,
            fontsize=9.0,
            fontfamily="monospace",
            color="#e2e8f0",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#1e293b", edgecolor="#10b981", lw=1.5, alpha=0.95)
        )
        ax6.set_title("6. Birleşik Modalite Modeli Özeti", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        
        plt.tight_layout(pad=2.5)
        plt.savefig(cikti_yolu, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"📊 [Visualizer] 6-Panelli Teşhis Panosu başarıyla kaydedildi: {cikti_yolu}")
