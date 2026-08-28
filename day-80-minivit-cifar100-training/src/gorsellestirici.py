"""
Mini Vision Transformer Eğitim ve Regülarizasyon Teşhis Panosu
--------------------------------------------------------------
6 panelli yüksek çözünürlüklü eğitim dinamikleri, Mixup/CutMix görselleştirmesi,
öğrenme oranı çizelgesi ve regülarizasyon ablasyon paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


class EgitimGorsellestirici:
    """
    MiniViT eğitim eğrilerini, veri artırma örneklerini ve regülarizasyon analizini görselleştiren sınıf.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_egitim_paneli(
        self,
        gecmis: Dict[str, List[float]],
        orijinal_gorseller: Tuple[np.ndarray, np.ndarray],
        mixup_gorsel: np.ndarray,
        cutmix_gorsel: np.ndarray,
        ablasyon_sonuclari: Dict[str, float],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı eğitim ve regülarizasyon teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 80: Sıfırdan MiniViT'in CIFAR-100 Üzerinde Eğitimi & Regülarizasyon Dinamikleri Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        epoklar = list(range(1, len(gecmis["egitim_kaybi"]) + 1))

        # -------------------------------------------------------------
        # PANEL 1: Eğitim & Doğrulama Kayıp Eğrileri
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epoklar, gecmis["egitim_kaybi"], "o-", color="#e53e3e", linewidth=2, label="Eğitim Kaybı (Mixup+LabelSmooth)")
        ax1.plot(epoklar, gecmis["dogrulama_kaybi"], "s--", color="#3182ce", linewidth=2, label="Doğrulama Kaybı (Temiz)")
        ax1.set_title("1. Eğitim & Doğrulama Kayıp Eğrileri", fontsize=12, fontweight="bold", color="#2c5282")
        ax1.set_xlabel("Epok", fontsize=10)
        ax1.set_ylabel("Kayıp (Loss)", fontsize=10)
        ax1.legend(loc="upper right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 2: Top-1 ve Top-5 Doğruluk (%) Dinamikleri
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epoklar, gecmis["dogrulama_top1_acc"], "o-", color="#38a169", linewidth=2.2, label="Val Top-1 Doğruluk (%)")
        ax2.plot(epoklar, gecmis["dogrulama_top5_acc"], "^--", color="#805ad5", linewidth=2.2, label="Val Top-5 Doğruluk (%)")
        ax2.plot(epoklar, gecmis["egitim_top1_acc"], ":", color="#a0aec0", linewidth=1.8, label="Eğitim Top-1 (%)")
        ax2.set_title("2. Top-1 ve Top-5 Doğruluk (%) Dinamikleri", fontsize=12, fontweight="bold", color="#22543d")
        ax2.set_xlabel("Epok", fontsize=10)
        ax2.set_ylabel("Doğruluk (%)", fontsize=10)
        ax2.legend(loc="lower right", frameon=True)

        # -------------------------------------------------------------
        # PANEL 3: Mixup & CutMix Veri Artırma Örnekleri
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.axis("off")
        
        # 2x2 alt ızgara oluştur
        img1, img2 = orijinal_gorseller
        # Boyutları (H, W, C) formatına getir
        def format_img(im):
            if im.shape[0] == 3:
                im = np.transpose(im, (1, 2, 0))
            return np.clip((im - im.min()) / (im.max() - im.min() + 1e-12), 0.0, 1.0)

        f_img1 = format_img(img1)
        f_img2 = format_img(img2)
        f_mix = format_img(mixup_gorsel)
        f_cut = format_img(cutmix_gorsel)

        # 4 görseli tek bir bileşik görsel olarak birleştir (2x2)
        h, w, _ = f_img1.shape
        grid_img = np.zeros((2 * h + 8, 2 * w + 8, 3))
        grid_img[0:h, 0:w] = f_img1
        grid_img[0:h, w+8:2*w+8] = f_img2
        grid_img[h+8:2*h+8, 0:w] = f_mix
        grid_img[h+8:2*h+8, w+8:2*w+8] = f_cut

        ax3.imshow(grid_img)
        ax3.set_title("3. Veri Artırma: [Üst: Orijinaller | Sol Alt: Mixup | Sağ Alt: CutMix]", fontsize=10.5, fontweight="bold", color="#d69e2e")

        # -------------------------------------------------------------
        # PANEL 4: Cosine Annealing LR & Gradyan Normu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4_twin = ax4.twinx()
        
        l1 = ax4.plot(epoklar, gecmis["ogrenme_oranlari"], "m-", linewidth=2.2, label="Öğrenme Oranı (LR)")
        l2 = ax4_twin.plot(epoklar, gecmis["gradyan_normlari"], "c--", linewidth=1.8, label="Kırpılmış Gradyan Normu")
        
        ax4.set_title("4. Linear Warmup + Cosine Annealing LR & Gradyan Normu", fontsize=12, fontweight="bold", color="#553c9a")
        ax4.set_xlabel("Epok", fontsize=10)
        ax4.set_ylabel("Öğrenme Oranı", color="m", fontsize=10)
        ax4_twin.set_ylabel("Gradyan Normu (Clip=1.0)", color="c", fontsize=10)
        
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax4.legend(lns, labs, loc="upper right")

        # -------------------------------------------------------------
        # PANEL 5: Regülarizasyon Ablasyon Karşılaştırması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        metodlar = list(ablasyon_sonuclari.keys())
        skorlar = list(ablasyon_sonuclari.values())
        renkler = ["#cbd5e0", "#90cdf4", "#f6ad55", "#68d391"]

        bars = ax5.bar(metodlar, skorlar, color=renkler, edgecolor="#2d3748", width=0.55)
        ax5.set_title("5. Regülarizasyon Ablasyon Karşılaştırması (Top-1 Doğruluk %)", fontsize=12, fontweight="bold", color="#2c5282")
        ax5.set_ylabel("Top-1 Doğruluk (%)", fontsize=10)
        ax5.set_ylim(0, max(skorlar) + 15)
        plt.setp(ax5.get_xticklabels(), rotation=15, ha="right", fontsize=9)

        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"%{yval:.1f}", ha="center", va="bottom", fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "       MİNİVİT EĞİTİM & REGÜLARİZASYON SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Mixup & CutMix ile aşırı uydurma (overfitting) tamamen engellenir.\n"
            "  • Cosine Annealing + Warmup ile kararlı ve pürüzsüz yakınsama.\n"
            "  • Decoupled Weight Decay ile pos_embed ve biaslar korunur.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Veri artırma ve yumuşak kayıp nedeniyle daha fazla epok ihtiyacı.\n"
            "  • Hiperparametre hassasiyeti (Alpha, LR min, Warmup süresi).\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • CIFAR-100 ve Tiny-ImageNet gibi küçük veri setlerinde ViT başarısı.\n"
            "  • LoRA fine-tuning ve Knowledge Distillation (DeiT) entegrasyonu.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Aşırı agresif CutMix'in etiket-görsel bağlamını bozabilmesi."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. ViT Eğitim Reçetesi SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
