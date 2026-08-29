"""
Day 123: Reflexion - Sözel Öz-Eleştiri (Self-Critique) ve Episodik Hafıza Ajanı Ana Akışı.
Algoritmik kodlama, birim test değerlendirmesi ve çok turlu hata giderme gösterimi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.degerlendirici import TestDurumu
from src.reflexion_ajani import ReflexionAjani
from src.gorsellestirici import ReflexionGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 123: Reflexion Agent - Verbal Reinforcement Learning & Episodic Self-Critique")
    print("=" * 105)

    # -------------------------------------------------------------
    # ADIM 1: Problem Tanımı ve Birim Testlerin Hazırlanması
    # -------------------------------------------------------------
    print("\n[1/3] Algoritmik Görev ve Birim Test Kümesi Başlatılıyor...")
    problem = (
        "Bir tamsayı dizisi verildiğinde, toplamı en büyük olan ardışık alt diziyi (Maximum Subarray Sum) "
        "bulan ve bu maksimum toplamı döndüren 'max_alt_dizi_toplami' fonksiyonunu yazın."
    )
    fonksiyon_adi = "max_alt_dizi_toplami"

    test_kumesi = [
        TestDurumu(girdi=[-2, 1, -3, 4, -1, 2, 1, -5, 4], beklenen=6, aciklama="Karışık Pozitif/Negatif"),
        TestDurumu(girdi=[-1, -2, -3, -4], beklenen=-1, aciklama="Tümü Negatif Sınır Durumu"),
        TestDurumu(girdi=[5, 4, -1, 7, 8], beklenen=23, aciklama="Pozitif Ağırlıklı Dizi"),
        TestDurumu(girdi=[1], beklenen=1, aciklama="Tek Elemanlı Dizi"),
    ]

    print(f"\n[-] HEDEF GÖREV:\n'{problem}'")
    print(f"[-] Fonksiyon Adı    : {fonksiyon_adi}")
    print(f"[-] Toplam Test Sayısı: {len(test_kumesi)} Birim Test\n")

    # -------------------------------------------------------------
    # ADIM 2: Reflexion Çok Turlu İteratif Hata Ayıklama
    # -------------------------------------------------------------
    print("[2/3] Reflexion Aktör-Değerlendirici-Reflector Döngüsü Yürütülüyor...")
    ajan = ReflexionAjani(maksimum_deneme=3)
    rapor = ajan.iteratif_hata_ayikla(problem, fonksiyon_adi, test_kumesi)

    print("\n" + "=" * 95)
    print("                      🔄 REFLEXION İTERATİF DENEME VE DERS GEÇMİŞİ                       ")
    print("=" * 95)
    for d in rapor["deneme_gecmisi"]:
        print(f"\n>>> DENEME {d['deneme_no']} | DURUM: {d['durum']} | ÖDÜL: {d['odul']*100:.1f}%")
        if d["hata"]:
            print(f"  [!] Hata Teşhisi : {d['hata']}")
        print(f"  [💡] Sözel Öz-Eleştiri: {d['oz_elestiri']}")
    print("-" * 95)

    print(f"\n[✓] NİHAİ ÇÖZÜM KODU:\n{rapor['nihai_kod']}")
    print(f"[✓] ÇÖZÜLDÜ MÜ       : {'EVET' if rapor['cozuldu'] else 'HAYIR'}")
    print(f"[✓] GEREKEN DENEME   : {rapor['toplam_deneme']} Deneme (Trial)")
    print(f"[✓] ÇÖZÜM SÜRESİ     : {rapor['toplam_sure_sn']*1000:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 3: Mimari Kıyaslama ve Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] Zero-Shot vs Reflexion Pass@k Kıyaslaması ve Teşhis Panosu Çiziliyor...")
    karsilastirma = ajan.benchmark_karsilastir()

    print("\n" + "=" * 90)
    print(f"{'YÖNTEM':<24} | {'PASS@K BAŞARI (%)':<20} | {'HATA TEKRARI (%)':<20}")
    print("-" * 90)
    for y, p, ht in zip(
        karsilastirma["denemeler"],
        karsilastirma["pass_oranlari"],
        karsilastirma["hata_tekrarlama_orani"],
    ):
        print(f"{y:<24} | %{p:>17.1f} | %{ht:>17.1f}")
    print("-" * 90)

    gorsellestirici = ReflexionGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "reflexion_ajan_paneli.png",
    )
    gorsellestirici.pano_olustur(
        calisma_raporu=rapor,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 123: REFLEXION OTONOM HATA GİDERME AJANI BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
