"""
Day 115: Model Merging (SLERP, TIES-Merging, DARE) ile Model Füzyonu Ana Akışı.
Sıfır GPU eğitimi ve geriye yayılım olmadan uzman modellerin parametre uzayında birleştirilmesi,
çok alanlı (Multi-Task) başarım kıyası ve 6 panelli teşhis panosu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.birlestirme_laboratuvari import BirlestirmeLaboratuvari
from src.gorsellestirici import ModelMergingGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 115: Model Merging (SLERP, TIES-Merging & DARE) Zero-Shot Model Fusion")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Laboratuvar ve Taban Model Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] Taban Model ve Çok Alanlı Test Ortamı Başlatılıyor...")
    lab = BirlestirmeLaboratuvari(in_dim=64, hidden_dim=128, out_dim=32, cihaz=cihaz)

    param_sayisi = sum(p.numel() for p in lab.taban_model.parameters())
    print(f"  * Taban Model Parametre Sayısı  : {param_sayisi:,}")
    print("  * Hedef Alanlar                 : Matematik, Kodlama, Genel Akıl Yürütme")
    print("  * Füzyon Eğitim Maliyeti        : 0 GPU Saati ($0 GPU Cost - Sıfır Geri Yayılım!)")

    # -------------------------------------------------------------
    # ADIM 2: Uzman Modellerin Eğitimi ve Model Füzyon Deneyleri
    # -------------------------------------------------------------
    print("\n[2/3] Uzman Modeller Eğitiliyor ve SLERP / TIES / DARE Birleştirmeleri Koşturuluyor...")
    sonuclar = lab.fuzyon_deneyini_kostur()

    print("\n--- MODEL MERGING ÇOK ALANLI BAŞARIM DEĞERLENDİRME TABLOSU ---")
    print(f"{'MODEL ADI':<24} | {'MATEMATİK':<12} | {'KODLAMA':<12} | {'GENEL AKIL':<12} | {'BİLEŞİK BAŞARI (%)':<20}")
    print("-" * 95)
    for model_adi, metrikler in sonuclar.items():
        print(
            f"{model_adi:<24} | "
            f"%{metrikler['Matematik Skoru']:>9.2f} | "
            f"%{metrikler['Kodlama Skoru']:>9.2f} | "
            f"%{metrikler['Genel Akıl Yürütme']:>9.2f} | "
            f"%{metrikler['Bileşik Başarı']:>17.2f}"
        )
    print("-" * 95)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Model Merging Teşhis Panosu Çiziliyor...")
    gorsellestirici = ModelMergingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "model_merging_paneli.png",
    )
    gorsellestirici.pano_olustur(
        sonuclar,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 115: Model Merging (SLERP, TIES, DARE) Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
