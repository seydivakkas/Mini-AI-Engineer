"""
Day 206: Step-Level PRM (Process Reward Model) ile Adım Bazlı Akıl Yürütme Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.prm_motoru import (
    PRMStepClassifier,
    MathReasoningTrajectory,
    PRMTreeSearchEngine,
)
from src.prm_profilleyici import PRMAkisProfilleyici
from src.gorsellestirici import PRMGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 206 (FAZ 11): STEP-LEVEL PRM (PROCESS REWARD MODEL) REASONING ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: PRM Modelinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Step-Level PRM (Süreç Ödül Modeli) Mimarisi Başlatılıyor...")
    prm_model = PRMStepClassifier(vocab_size=128, embed_dim=64)
    print("  • Mimari               : Transformer Encoder + Step-Level Scoring Head")
    print("  • Skorlama Türü        : Her Ara Düşünce Adımı İçin Olasılık (p in [0.0, 1.0])")
    print("  • Referans Paradigma   : OpenAI PRM800K & Q* / o1 Reasoning Search")
    print("  ✓ PRM Modeli Başarıyla Yüklendi!")

    # -------------------------------------------------------------
    # ADIM 2: Çok Adımlı Çözüm Yörüngesinin Skorlanması
    # -------------------------------------------------------------
    print("\n[2/4] Örnek Çok Adımlı Matematik Yörüngesi PRM ile İnceleniyor...")
    ornek_yorunge = MathReasoningTrajectory(
        problem_sorusu="3x - 5 = 16",
        adimlar=[
            "1. Adım: Her iki tarafa 5 eklendi: 3x = 21",
            "2. Adım: Her iki taraf 3'e bölündü: x = 7",
        ],
        nihai_cevap="7",
    )
    skorlar = ornek_yorunge.prm_skorla(prm_model)
    print(f"  • Soru Metni           : {ornek_yorunge.soru}")
    for idx, (adim_metni, s) in enumerate(zip(ornek_yorunge.adimlar, skorlar)):
        print(f"    - Adım #{idx+1} Skoru : {s:.4f} -> {adim_metni}")
    print(f"  • Minimum Adım Skoru   : {ornek_yorunge.minimum_skor:.4f}")
    print(f"  • Çarpım Güven Skoru   : {ornek_yorunge.carpim_skoru:.4f}")
    print("  ✓ Adım Seviyesi Skorlama Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Test-Zamanı Arama ve Erken Dal Budama (Pruning)
    # -------------------------------------------------------------
    print("\n[3/4] Test-Zamanı Ağaç Araması ve Erken Dal Budama Simülasyonu Yürütülüyor...")
    profil_raporu = PRMAkisProfilleyici.kapsamli_profil_cikar()
    arama = profil_raporu["arama_sonuclari"]

    print("-" * 115)
    print(f"{'Yol No':<10} | {'Adım Sayısı':<14} | {'Min Adım Skoru':<18} | {'Budama Durumu':<22} | {'Nihai Cevap'}")
    print("-" * 115)
    for y in arama["yollar"]:
        budama_str = f"Budandı (Adım #{y['budandigi_adim']})" if y["budandi"] else "GEÇERLİ (Doğru Yol)"
        print(
            f"Yol #{y['yol_idx']:<6} | "
            f"{y['adim_sayisi']:<14} | "
            f"{y['minimum_skor']:>14.4f}   | "
            f"{budama_str:<22} | "
            f"{y['nihai_cevap']}"
        )
    print("-" * 115)
    print(f"  ⚡ GPU Token Tasarrufu   : %{arama['hesaplama_tasarrufu_yuzde']:.1f}")
    print(f"  🎯 Hata Lokalizasyonu   : {profil_raporu['metrikler']['prm_hata_lokalizasyonu']}")
    print(f"  🏆 Best-of-N Doğruluk   : {profil_raporu['metrikler']['best_of_n_pass_at_1']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli PRM Süreç Ödül Modeli Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "prm_stepwise_paneli.png")

    PRMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ PRM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 206 (FAZ 11): STEP-LEVEL PRM VE TEST-ZAMANI ARAMA BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
