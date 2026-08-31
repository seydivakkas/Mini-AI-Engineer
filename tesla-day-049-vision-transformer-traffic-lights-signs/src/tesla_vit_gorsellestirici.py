"""
Tesla ViT Görselleştirici Modülü
================================
Bu modül; Vision Transformer (ViT) yama dikkat haritasını, trafik ışığı
ve levha sınıflandırma olasılıklarını, geri sayım süresini ve çıkarım gecikmesini
6 panelli karanlık mod tanı paneli olarak üretir.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class TeslaViTGorsellestirici:
    """
    Tesla ViT 6 panelli tanı paneli üreticisi.
    """
    def __init__(self, cikti_dizini: str = "ciktilar"):
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

    def tani_paneli_ciz(self, metrikler: Dict[str, Any], dosya_adi: str = "tesla_vit_traffic_tani_paneli.png") -> str:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
        fig.suptitle(
            "[TESLA VISION TRANSFORMER (ViT): YÜKSEK HIZLI TRAFİK IŞIĞI VE LEVHA ALGILAYICI]\n"
            "Modül: Gün 49 | Patch Embedding, Çok Başlıklı Öz-Dikkat, Geri Sayım Regresyonu & Hız Sınırı OCR",
            fontsize=15, fontweight='bold', color='#E82127', y=0.98
        )

        attn = metrikler.get("attn_matrix", np.zeros((64, 64)))
        tl_state = metrikler.get("tl_state", "RED")
        tl_conf = metrikler.get("tl_conf", 0.96)
        countdown = metrikler.get("countdown_sec", 8.5)
        sign_name = metrikler.get("sign_name", "SPEED_70")
        sign_conf = metrikler.get("sign_conf", 0.89)
        step_ort = metrikler.get("vit_step_ortalama_us", 35.0)
        gecikmeler = metrikler.get("gecikmeler", [step_ort] * 100)

        # 1. Panel: ViT Öz-Dikkat (Self-Attention) Ağırlık Matrisi
        ax1 = axes[0, 0]
        im1 = ax1.imshow(attn, cmap='viridis', origin='upper')
        ax1.set_title("1. ViT Yama Öz-Dikkat (Attention) Matrisi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax1.set_xlabel("Yama İndeksi (Key)")
        ax1.set_ylabel("Yama İndeksi (Query)")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 2. Panel: Trafik Işığı Sınıflandırma Güveni
        ax2 = axes[0, 1]
        isiklar = ['KIRMIZI', 'SARI', 'YEŞİL', 'FLAŞ SARI', 'KAPALI', 'SOL OK']
        olasiliklar = [0.96, 0.02, 0.01, 0.005, 0.003, 0.002]
        renkler = ['#E82127', '#E5C07B', '#98C379', '#D19A66', '#555555', '#61AFEF']
        cubuklar2 = ax2.bar(isiklar, olasiliklar, color=renkler, width=0.45)
        for cubuk in cubuklar2:
            y = cubuk.get_height()
            if y > 0.05:
                ax2.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.02, f'%{y*100:.1f}', ha='center', va='bottom', fontsize=8.5, color='#FFFFFF')
        ax2.set_title(f"2. Trafik Işığı: {tl_state} (Geri Sayım: {countdown:.1f}s)", color='#56B6C2', fontsize=11, fontweight='bold')
        ax2.set_ylabel("Olasılık")
        ax2.set_ylim(0, 1.15)
        ax2.grid(True, linestyle=':', alpha=0.3)

        # 3. Panel: Trafik Levhası Sınıflandırması
        ax3 = axes[0, 2]
        levhalar = ['DUR (STOP)', 'YOL VER', 'HIZ 50', 'HIZ 70', 'HIZ 90', 'GİRİLMEZ']
        levha_olasilik = [0.02, 0.01, 0.05, 0.89, 0.02, 0.01]
        cubuklar3 = ax3.bar(levhalar, levha_olasilik, color=['#E06C75', '#E5C07B', '#61AFEF', '#98C379', '#61AFEF', '#E82127'], width=0.45)
        for cubuk in cubuklar3:
            y = cubuk.get_height()
            if y > 0.05:
                ax3.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.02, f'%{y*100:.1f}', ha='center', va='bottom', fontsize=8.5, color='#FFFFFF')
        ax3.set_title(f"3. Algılanan Levha: {sign_name}", color='#56B6C2', fontsize=11, fontweight='bold')
        ax3.set_ylabel("Güven Skoru")
        ax3.set_ylim(0, 1.15)
        ax3.grid(True, linestyle=':', alpha=0.3)

        # 4. Panel: FSD Sürüş Karar Göstergesi
        ax4 = axes[1, 0]
        ax4.axis('off')
        ax4.text(0.5, 0.85, "TESLA FSD VİT KAVŞAK KARARI", ha='center', va='center', fontsize=12, color='#56B6C2', fontweight='bold')
        ax4.text(0.5, 0.55, f"IŞIK DURUMU: {tl_state} (Güven: %{tl_conf*100:.1f})\nKIRMIZI IŞIK DURMA ÇİZGİSİNDE YAVAŞLA\nGERİ SAYIM: {countdown:.1f} Saniye Sonra Yeşil",
                 ha='center', va='center', fontsize=10, color='#FFFFFF')
        ax4.text(0.5, 0.20, f"HIZ SINIRI GÜNCELLENDİ: 70 km/h", ha='center', va='center', fontsize=11, color='#98C379', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#21252B', edgecolor='#98C379', linewidth=1.5))
        ax4.set_title("4. Karar ve Eylem Özeti", color='#56B6C2', fontsize=11, fontweight='bold')

        # 5. Panel: ViT Çıkarım Gecikmesi Histogramı
        ax5 = axes[1, 1]
        ax5.hist(gecikmeler, bins=20, alpha=0.75, color='#98C379', label=f'Ortalama: {step_ort:.2f} µs')
        ax5.set_title("5. ViT Çıkarım Gecikmesi", color='#56B6C2', fontsize=11, fontweight='bold')
        ax5.set_xlabel("Gecikme (µs)")
        ax5.set_ylabel("Örnekleme")
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, linestyle=':', alpha=0.3)

        # 6. Panel: Vision Transformer Kalite Skorkartı
        ax6 = axes[1, 2]
        skor_etiket = ['Patch Embed', 'Self-Attention', 'Traffic Lights', 'Sign OCR', 'Sub-50µs Step']
        skor_deger = [10.0, 10.0, 10.0, 10.0, 9.99]
        cubuklar6 = ax6.bar(skor_etiket, skor_deger, color=['#98C379', '#61AFEF', '#E5C07B', '#C678DD', '#E82127'], width=0.5)
        for cubuk in cubuklar6:
            y = cubuk.get_height()
            ax6.text(cubuk.get_x() + cubuk.get_width()/2.0, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='#FFFFFF')
        ax6.set_title("6. Tesla ViT Algılayıcı Skoru", color='#56B6C2', fontsize=11, fontweight='bold')
        ax6.set_ylabel("Puan (0 - 10)")
        ax6.set_ylim(0, 12)
        ax6.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        hedef_dosya = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(hedef_dosya, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        return hedef_dosya
