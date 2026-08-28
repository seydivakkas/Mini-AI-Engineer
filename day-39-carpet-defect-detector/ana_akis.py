"""
Day 39: Halı Dokuma Hataları, Leke ve Kusur Tespiti Ana Yürütme Betiği.
"""

import os
from PIL import Image
from src.sentetik_kusur_uretici import SentetikKusurluHaliUretici
from src.anomali_tespitci import AnomaliTespitci
from src.morfolojik_filtre import MorfolojikKusurFiltresi
from src.kontur_analizci import KonturAnalizci
from src.kusur_siniflandirici import KusurSiniflandirici
from src.gorsellestirici import HaliKusurGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Referans ve Kusurlu Halı Görsellerinin Hazırlanması")
    print("=" * 80)

    referans_gorseli = SentetikKusurluHaliUretici.temiz_referans_uret(400, 300)
    test_gorseli, enjekte_edilenler = SentetikKusurluHaliUretici.kusurlu_test_uret(referans_gorseli)

    print(f"[+] Referans Altın Numune Üretildi: 400x300 Piksel")
    print(f"[+] Test Numunesine {len(enjekte_edilenler)} Adet Dokuma Hatası Enjekte Edildi:")
    for h in enjekte_edilenler:
        print(f"    - {h['tur']:<20} | Konum: {h['konum']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Kalıntı Haritası, Pikselsel Anomali Tespiti ve Morfolojik Filtreleme")
    print("=" * 80)

    anomali_tespitci = AnomaliTespitci(sigma_filtre=2.5, esik_carpani=2.8)
    anomali_sonuc = anomali_tespitci.anomali_haritasi_cikar(test_gorseli, referans_gorseli=referans_gorseli)
    print(f"[+] Adaptif İstatiksel Eşik Değeri: {anomali_sonuc['esik_degeri']}")

    morfoloji = MorfolojikKusurFiltresi(acma_iter=1, kapama_iter=2, min_piksel_alani=25)
    temiz_maske = morfoloji.temizle_ve_birlestir(anomali_sonuc["ham_maske"])
    print(f"[+] Morfolojik Açma/Kapama Tamamlandı (Gürültü Pikselleri Elendi)")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: Bağlantılı Bileşen, Kontur Geometrisi ve Kusur Sınıflandırma")
    print("=" * 80)

    kontur_analizci = KonturAnalizci(min_kusur_alani=30)
    ham_kusurlar = kontur_analizci.analiz_et(temiz_maske)

    siniflandirilmis_kusurlar = [KusurSiniflandirici.kusuru_siniflandir(k) for k in ham_kusurlar]
    parti_raporu = KusurSiniflandirici.parti_kalite_degerlendir(siniflandirilmis_kusurlar)

    print(f"[+] Tespit Edilen Kusur Sayısı: {len(siniflandirilmis_kusurlar)} Adet")
    print("\n[+] DETAYLI KUSUR ANALİZ TABLOSU:")
    print(f"{'Kusur ID':<10} | {'Kusur Türü':<18} | {'Alan (px)':<10} | {'En-Boy (AR)':<12} | {'Dairesellik':<12} | {'Şiddet':<12} | {'Üretim Aksiyonu'}")
    print("-" * 110)

    for k in siniflandirilmis_kusurlar:
        print(f"{k['kusur_id']:<10} | {k['kusur_turu']:<18} | {k['alan']:<10} | {k['en_boy_orani']:<12.2f} | {k['dairesellik']:<12.2f} | {k['siddet']:<12} | {k['uretim_aksiyonu']}")

    print("\n[+] FABRİKA KALİTE KONTROL ÖZETİ:")
    print(f"    - Genel Parti Kalite Kararı : {parti_raporu['parti_kalite_karari']}")
    print(f"    - Toplam Kusurlu Alan (px)  : {parti_raporu['toplam_kusurlu_alan_piksel']} px")
    print(f"    - Kritik Kusur Sayısı       : {parti_raporu['kritik_kusur_sayisi']}")
    print(f"    - Sevkiyat Onayı            : {'ONAYLANDI (PASS)' if parti_raporu['parti_onayi'] else 'REDDEDİLDİ (FAIL)'}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: 6 Panelli Kalite Kontrol Teşhis Panosunun Kaydedilmesi")
    print("=" * 80)

    cikis_resmi = HaliKusurGorsellestirici.kusur_paneli_ciz(
        test_gorseli=test_gorseli,
        anomali_haritasi=anomali_sonuc["anomali_skor_haritasi"],
        temiz_maske=temiz_maske,
        tespit_edilen_kusurlar=siniflandirilmis_kusurlar,
        parti_raporu=parti_raporu,
        hedef_path="day-39-carpet-defect-detector/ciktilar/hali_kusur_tespit_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 39: HALI DOKUMA HATALARI VE KUSUR TESPİTİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
