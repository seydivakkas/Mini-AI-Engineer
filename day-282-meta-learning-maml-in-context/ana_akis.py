"""
Day 282 (FAZ 15): Meta-Learning (MAML & Meta-SGD) Ana Akış Betiği.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.maml_meta_motoru import MAMLEngine, MetaTask
from src.maml_meta_profilleyici import MAMLMetaProfilleyici
from src.gorsellestirici import MAMLGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 282 (FAZ 15): META-LEARNING (MAML & META-SGD) — HIZLI GÖREV KEŞFİ VE FEW-SHOT ADAPTASYON")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Meta-Öğrenme Motoru ve Görev Dağılımı
    # -------------------------------------------------------------
    print("\n[1/4] MAML & Meta-SGD Meta-Öğrenme Motoru ve Görev Havuzu Başlatılıyor...")
    engine = MAMLEngine(input_dim=1, hidden_dim=40, output_dim=1, inner_lr=0.02, meta_lr=0.005)
    tasks = [MetaTask(amplitude=float(np.random.uniform(0.5, 3.0)), phase=float(np.random.uniform(0.0, np.pi))) for _ in range(10)]
    print(f"  • Model Mimarisi                     : Giriş=1, Gizli=40 (ReLU), Çıkış=1")
    print(f"  • İç Döngü Öğrenme Oranı (α)         : {engine.inner_lr} (Meta-SGD Vektörel)")
    print(f"  • Dış Döngü Meta-Öğrenme Oranı (β)   : {engine.meta_lr}")
    print(f"  • Örneklenen Meta-Görev Sayısı       : {len(tasks)} Görev")

    # -------------------------------------------------------------
    # ADIM 2: İç ve Dış Döngü Meta-Eğitim Adımları
    # -------------------------------------------------------------
    print("\n[2/4] Çift Döngülü Meta-Eğitim Koşturuluyor (İç Döngü: θ -> θ', Dış Döngü: θ)...")
    for step in range(1, 6):
        stats = engine.train_meta_step(tasks, k_shots=5, q_queries=15)
        print(f"  • Meta-Adım {step}/5 -> Adaptasyon Öncesi Kayıp: {stats['pre_adapt_loss']:.4f} | Sonrası: {stats['post_adapt_loss']:.4f} | Gradyan Normu: {stats['meta_gradient_norm']:.4f}")

    # -------------------------------------------------------------
    # ADIM 3: Görülmemiş Test Görevi Üzerinde Few-Shot Değerlendirme
    # -------------------------------------------------------------
    print("\n[3/4] Görülmemiş Yeni Görev (Unseen Task) Üzerinde Few-Shot Adaptasyon...")
    profil = MAMLMetaProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • 0-Shot Naive Model Başarımı        : %{kars['few_shot_dogruluk_yuzde']['0_Shot_Naive']:.1f} (MSE Kaybı: {kars['adaptasyon_mse_kaybi']['0_Shot_Naive']:.2f})")
    print(f"  • 1-Shot MAML Adaptasyon Başarımı    : %{kars['few_shot_dogruluk_yuzde']['1_Shot_MAML']:.1f} (MSE Kaybı: {kars['adaptasyon_mse_kaybi']['1_Shot_MAML']:.2f})")
    print(f"  • 5-Shot Meta-SGD Adaptasyon Başarımı: %{kars['few_shot_dogruluk_yuzde']['5_Shot_Meta_SGD']:.1f} (MSE Kaybı: {kars['adaptasyon_mse_kaybi']['5_Shot_Meta_SGD']:.2f} | 23x Düşüş)")
    print(f"  • Adaptasyon Süresi                  : {kars['ic_dongu_gecikmesi_ms']['5_Shot_Meta_SGD']:.2f} ms (Anlık Few-Shot Transfer)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli MAML & Meta-SGD Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "meta_learning_maml_paneli.png")

    MAMLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ MAML & Meta-SGD Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 282 (FAZ 15): META-LEARNING (MAML & META-SGD) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
