"""
Day 158: Büyük Akıl Yürüten Modelin (DeepSeek-R1) Düşünce İncilerini Küçük Modele Damıtma (Reasoning Trace Distillation) Ana Akışı.
Öğretmen İzi Üretimi -> Kalite Filtreleme -> SFT Damıtma -> Benchmark Kıyası.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ogretmen_model_simulasyonu import OgretmenModelSimulasyonu
from src.iz_filtreleyici import DusunceIziFiltreleyici
from src.damitma_egitici import DamitmaEgitici
from src.gorsellestirici import DamitmaGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 158: Reasoning Trace Distillation & SFT Thought Transfer Pipeline (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. ÖĞRETMEN DÜŞÜNCE İZLERİ VE FİLTRELEME
    # -------------------------------------------------------------
    print("\n[1/3] Öğretmen Modelden (DeepSeek-R1 671B) Ham Düşünce İzleri Toplanıyor & Filtreleniyor...")
    senaryolar = ["mukemmel", "dongulu_hatali", "hatali_sonuc"]
    ornek_izler = []

    for s in senaryolar:
        iz = OgretmenModelSimulasyonu.iz_uret("3x + 15 = 45 denkleminin kökünü bulunuz.", senaryo=s)
        filtre = DusunceIziFiltreleyici.izi_degerlendir(iz)
        ornek_izler.append((iz, filtre))
        durum = "✓ KABUL EDİLDİ" if filtre["kabul_edildi_mi"] else f"✗ REDDEDİLDİ ({filtre['red_nedeni']})"
        print(f"  • Senaryo: {s:<15} | Kalite Skoru: {filtre['kalite_skoru']:.2f} | Durum: {durum}")

    mukemmel_iz = ornek_izler[0][0]

    # -------------------------------------------------------------
    # 2. ÖĞRENCİ MODEL SFT DAMITMA EĞİTİMİ
    # -------------------------------------------------------------
    print("\n[2/3] Küçük Öğrenci Model (Qwen-1.5B) Kürate Edilmiş Düşünce İzleriyle Eğitiliyor (SFT)...")
    egitim_sonucu = DamitmaEgitici.egitimi_simule_et(filtrelenmis_ornek_sayisi=1000)

    print("\n" + "-" * 85)
    print(f"{'Adım':<10} | {'SFT Kaybı (Loss)':<20} | {'1.5B MATH Doğruluğu (%)':<25}")
    print("-" * 85)
    for adim, kayip, acc in zip(egitim_sonucu["adimlar"], egitim_sonucu["kayiplar"], egitim_sonucu["ogrenci_dogruluk_egrisi"]):
        print(f"{adim:<10} | {kayip:<20.2f} | %{acc:<25.1f}")
    print("-" * 85)

    print("\nBENCHMARK VE KAPASİTE TRANSFERİ RAPORU:")
    for model_adi, metrikler in egitim_sonucu["benchmark_kiyasi"].items():
        print(f"  • {model_adi:<42}: MATH = %{metrikler['math_dogruluk']:<5.1f} | GSM8K = %{metrikler['gsm8k_dogruluk']:<5.1f} | Boyut = {metrikler['parametre_boyutu']}")
    print(f"\n  >> Performans Sıçraması: +%{egitim_sonucu['performans_kazanci_yuzde']} MATH Artışı")
    print(f"  >> Öğretmen Seviyesi  : %{egitim_sonucu['ogretmen_yakalama_orani']} oranında yakalandı!")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Reasoning Trace Distillation Teşhis Panosu Üretiliyor...")
    gorsellestirici = DamitmaGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "reasoning_trace_distillation_paneli.png")
    gorsellestirici.pano_olustur(egitim_sonucu, mukemmel_iz, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 158: REASONING TRACE DISTILLATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
