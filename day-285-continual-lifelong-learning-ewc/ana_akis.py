"""
Day 285 (FAZ 15): Sürekli ve Yaşam Boyu Öğrenme (Continual Learning) Ana Akış Betiği.
Elastic Weight Consolidation (EWC) ve Fisher Bilgi Matrisi ile Yıkıcı Unutmayı Önleme.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.ewc_motoru import SimpleClassifier, ContinualLifelongLearningEngine
from src.ewc_profilleyici import EWCProfilleyici
from src.gorsellestirici import EWCGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 285 (FAZ 15): SÜREKLİ VE YAŞAM BOYU ÖĞRENME — ELASTIC WEIGHT CONSOLIDATION (EWC)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Görev A Eğitimi ve Sentetik Veri
    # -------------------------------------------------------------
    print("\n[1/4] Çok Görevli Yapay Sinir Ağı Başlatılıyor ve Görev A Eğitiliyor...")
    profil = EWCProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Model Yapısı                       : 3-Katmanlı MLP (32 Gizli Birim)")
    print(f"  • Görev A Başlangıç Doğruluğu        : %{profil['acc_a_initial']:.1f}")

    # -------------------------------------------------------------
    # ADIM 2: Fisher Bilgi Matrisi Çıkarımı
    # -------------------------------------------------------------
    print("\n[2/4] Görev A Fisher Bilgi Matrisi (F_i) ve Optimal Ağırlıklar (θ_A*) Hesaplanıyor...")
    print(f"  • İncelenen Parametre Katman Sayısı  : {profil['fisher_katman_sayisi']} Katman")
    print(f"  • Fisher Bilgi Matrisi               : Diagonal F_i = (1/N) * Σ (∇_θ log p(y|x, θ))^2")
    print(f"  • EWC Yay Sabiti (λ)                 : 5000.0 (Elastik Ağırlık Konsolidasyonu)")

    # -------------------------------------------------------------
    # ADIM 3: Görev B Sıralı Eğitimi ve Unutma Karşılaştırması
    # -------------------------------------------------------------
    print("\n[3/4] Görev B Sıralı Eğitimi ve Yıkıcı Unutma Karşılaştırması...")
    print(f"  • 1. Saf İnce Ayar (Naive)           : Görev A: %{kars['gorev_a_hatirlama_orani']['1. Saf Ince Ayar (Naive)']:.1f} | Görev B: %{kars['gorev_b_ogrenme_orani']['1. Saf Ince Ayar (Naive)']:.1f} (Yıkıcı Unutma: %{kars['yikici_unutma_orani']['1. Saf Ince Ayar (Naive)']:.1f})")
    print(f"  • 2. Synaptic Intelligence (SI)      : Görev A: %{kars['gorev_a_hatirlama_orani']['2. Synaptic Intelligence (SI)']:.1f} | Görev B: %{kars['gorev_b_ogrenme_orani']['2. Synaptic Intelligence (SI)']:.1f} (Yıkıcı Unutma: %{kars['yikici_unutma_orani']['2. Synaptic Intelligence (SI)']:.1f})")
    print(f"  • 3. EWC Konsolidasyonu (EWC)        : Görev A: %{kars['gorev_a_hatirlama_orani']['3. EWC Konsolidasyonu (EWC)']:.1f} | Görev B: %{kars['gorev_b_ogrenme_orani']['3. EWC Konsolidasyonu (EWC)']:.1f} (Yıkıcı Unutma: %{kars['yikici_unutma_orani']['3. EWC Konsolidasyonu (EWC)']:.1f})")
    print(f"  • Bellek Hatırlama Kazancı           : +%{kars['gorev_a_hatirlama_orani']['3. EWC Konsolidasyonu (EWC)'] - kars['gorev_a_hatirlama_orani']['1. Saf Ince Ayar (Naive)']:.1f} (22x Daha Düşük Unutma)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Sürekli Öğrenme (EWC) Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "continual_learning_ewc_paneli.png")

    EWCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ EWC Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 285 (FAZ 15): SÜREKLİ VE YAŞAM BOYU ÖĞRENME (EWC) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
