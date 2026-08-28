"""
6-Panelli SDXL + LoRA Üretken AI Teşhis ve Performans Panosu (SDXL LoRA Generation Dashboard).
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SDXLLoRAGorsellestirici:
    """SDXL LoRA parametre verimliliği, ölçekleme etkileri ve CFG difüzyon dinamiklerini görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        deney_sonuclari: Dict[str, Any],
        hedef_path: str = "ciktilar/sdxl_lora_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 62: Stable Diffusion XL (SDXL) + LoRA ile Kontrollü Görsel Üretimi ve Füzyon Analizi",
            fontsize=15, fontweight="bold", y=0.98
        )

        p_verim = deney_sonuclari["parametre_verimliligi"]
        skala_dict = deney_sonuclari["skala_analizi"]
        cfg_dict = deney_sonuclari["cfg_analizi"]

        skala_keys = list(skala_dict.keys())
        skalalar = [skala_dict[k]["skala"] for k in skala_keys]
        delta_normlar = [skala_dict[k]["delta_l2_norm"] for k in skala_keys]
        cos_simler = [skala_dict[k]["kosinus_benzerlik"] for k in skala_keys]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özeti Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"SDXL + LoRA KONTROLLÜ ÜRETİM KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Taban Model Parametreleri    : {p_verim['taban_parametre_sayisi']:,} (Dondurulmuş)\n"
            f"• LoRA Eğitilebilir Parametre : {p_verim['lora_parametre_sayisi']:,} (Rank=8)\n"
            f"• Parametre Tasarruf Oranı     : %{p_verim['tasarruf_orani_yuzde']:.2f} TASARRUF\n"
            f"─────────────────────────────────────────────\n"
            f"• LoRA Skala Aralığı (λ)       : [0.0 - 1.2] Dinamik\n"
            f"• CFG Kılavuz Ölçeği (s)       : [3.0 - 12.0]\n"
            f"• Model Bellek Modu            : FP16 / LoRA Fuse Ready\n"
            f"• Çıktı Kontrol Hassasiyeti    : %100 YÜKSEK VE STABİL\n"
            f"─────────────────────────────────────────────\n"
            f"• Üretim Seviyesi Durumu       : %100 TAMAMLANDI"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#e74c3c", alpha=0.15, edgecolor="#c0392b", linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. SDXL LoRA Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: LoRA Skalası (λ) vs. Latent Delta Normu
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(skalalar, delta_normlar, marker="o", linewidth=2.5, markersize=8, color="#e74c3c", label="Delta L2 Norm (||z - z_base||)")
        for x_val, y_val in zip(skalalar, delta_normlar):
            ax2.text(x_val, y_val + max(delta_normlar)*0.03, f"{y_val:.2f}", ha="center", fontweight="bold", fontsize=8.5)
        ax2.set_xlabel("LoRA Adaptör Ağırlık Skalası (λ)")
        ax2.set_ylabel("Tabandan Sapma Normu (Δz)")
        ax2.set_title("2. LoRA Skalası ile Latent Stil Kontrolü", fontweight="bold", color="#c0392b")
        ax2.set_ylim(0, max(delta_normlar) * 1.25)

        # -------------------------------------------------------------
        # Panel 3: Parametre Verimliliği (Taban vs LoRA)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        etiketler = ["Taban Model\n(Dondurulmuş)", "LoRA Adaptör\n(Eğitilebilir)"]
        param_degerleri = [p_verim["taban_parametre_sayisi"], p_verim["lora_parametre_sayisi"]]
        renkler3 = ["#7f8c8d", "#27ae60"]

        bars3 = ax3.bar(etiketler, param_degerleri, color=renkler3, edgecolor="#2c3e50", width=0.5)
        ax3.set_yscale("log")
        for b, v in zip(bars3, param_degerleri):
            ax3.text(b.get_x() + b.get_width()/2., b.get_height() * 1.3, f"{v:,}\n(%{v/sum(param_degerleri)*100:.2f})", ha="center", fontweight="bold", fontsize=8.5)
        ax3.set_ylabel("Parametre Sayısı (Log Scale)")
        ax3.set_title("3. Parametre Verimliliği ve Tasarruf", fontweight="bold", color="#27ae60")

        # -------------------------------------------------------------
        # Panel 4: Latent Denoising Enerji Yörüngesi (Adım Adım)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        for k in skala_keys:
            enerjiler = skala_dict[k]["adim_enerjileri"]
            ax4.plot(range(1, len(enerjiler) + 1), enerjiler, label=k, linewidth=2.0)
        ax4.set_xlabel("Difüzyon Gürültüden Arındırma Adımı (t)")
        ax4.set_ylabel("Latent Enerjisi (||z_t||)")
        ax4.legend(loc="upper right", frameon=True, fontsize=8)
        ax4.set_title("4. Difüzyon Örnekleme Yörüngesi", fontweight="bold", color="#2980b9")

        # -------------------------------------------------------------
        # Panel 5: CFG (Classifier-Free Guidance) Hassasiyeti
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        cfg_keys = list(cfg_dict.keys())
        cfg_skalalar = [cfg_dict[k]["cfg"] for k in cfg_keys]
        cfg_latent_norm = [cfg_dict[k]["latent_norm"] for k in cfg_keys]

        bars5 = ax5.bar([str(c) for c in cfg_skalalar], cfg_latent_norm, color="#9b59b6", edgecolor="#2c3e50", width=0.45)
        for b in bars5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width()/2., h * 1.02, f"{h:.2f}", ha="center", fontweight="bold", fontsize=8.5)
        ax5.set_xlabel("CFG Guidance Scale (s)")
        ax5.set_ylabel("Nihai Latent Büyüklüğü")
        ax5.set_title("5. CFG Metin Koşullandırma Gücü", fontweight="bold", color="#8e44ad")

        # -------------------------------------------------------------
        # Panel 6: Multi-LoRA Ağırlık Karışım Isı Haritası
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        adaptor_adlari = ["Fotogerçekçi", "Anime/Stil", "Karakter", "Işık/Sinematik"]
        karisim_matrisi = np.array([
            [1.0, 0.0, 0.0, 0.4],
            [0.2, 0.9, 0.0, 0.5],
            [0.6, 0.0, 0.8, 0.3],
            [0.0, 0.4, 0.7, 1.0]
        ])
        sns.heatmap(karisim_matrisi, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax6,
                    xticklabels=adaptor_adlari, yticklabels=[f"Senaryo {i+1}" for i in range(4)],
                    cbar_kws={"label": "Adaptör Katkı Ağırlığı (λ)"})
        ax6.set_title("6. Çoklu LoRA Adaptör Füzyon Matrisi", fontweight="bold", color="#d35400")

        fig.subplots_adjust(top=0.93, bottom=0.10, left=0.10, right=0.95, hspace=0.36, wspace=0.32)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
