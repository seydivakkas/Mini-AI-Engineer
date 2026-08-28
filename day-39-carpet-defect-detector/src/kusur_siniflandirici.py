"""
Dokuma Kusurları Sınıflandırıcı ve Şiddet Derecelendirme Motoru (Defect Classifier & QC Protocol).
"""

from typing import List, Dict, Any


class KusurSiniflandirici:
    """Geometrik ve yoğunluk metriklerine göre halı dokuma kusurunu sınıflandırır ve fabrika protokolünü belirler."""

    @classmethod
    def kusuru_siniflandir(cls, kusur_metrikleri: Dict[str, Any]) -> Dict[str, Any]:
        alan = kusur_metrikleri["alan"]
        ar = kusur_metrikleri["en_boy_orani"]
        dairesellik = kusur_metrikleri["dairesellik"]

        # Kural Tabanlı Endüstriyel Sınıflandırma
        if ar >= 3.2:
            tur = "IPLIK_KOPMASI"
            aciklama = "Çözgü/Atkı Yönünde İplik Kaçığı veya Kopması (Elongated Run)"
        elif alan >= 400 and (ar >= 1.8 or dairesellik < 0.35):
            tur = "DELIK_YIRTIK"
            aciklama = "Dokuma Bütünlüğü Bozulmuş Delik veya Yırtık (Hole/Tear)"
        elif dairesellik >= 0.45 and ar < 2.0:
            tur = "YAG_BOYA_LEKESI"
            aciklama = "Bölgesel Yağ Damlaması veya Boya Lekesi (Oil/Dye Stain)"
        else:
            tur = "DUGUM_TOPAKLANMA"
            aciklama = "İplik Topaklanması veya Dokuma Düğüm Hatası (Slub/Knot)"

        # Şiddet Derecelendirmesi (Severity Rating)
        if tur == "DELIK_YIRTIK" or alan >= 500:
            siddet = "KRITIK"
            aksiyon = "HATTI_DURDUR_HURDA_AYIR"
        elif tur in ["IPLIK_KOPMASI", "YAG_BOYA_LEKESI"] or alan >= 180:
            siddet = "ORTA_KUSUR"
            aksiyon = "IKINCI_KALITE_AYIR"
        else:
            siddet = "KUCUK_KUSUR"
            aksiyon = "REWORK_DUZELTME_ISTASYONU"

        sonuc = dict(kusur_metrikleri)
        sonuc.update({
            "kusur_turu": tur,
            "tur_aciklamasi": aciklama,
            "siddet": siddet,
            "uretim_aksiyonu": aksiyon
        })
        return sonuc

    @classmethod
    def parti_kalite_degerlendir(cls, siniflandirilmis_kusurlar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tüm tespit edilen kusurlardan genel parti kalite kararını çıkarır."""
        toplam_kusur = len(siniflandirilmis_kusurlar)
        kritik_sayisi = sum(1 for k in siniflandirilmis_kusurlar if k["siddet"] == "KRITIK")
        orta_sayisi = sum(1 for k in siniflandirilmis_kusurlar if k["siddet"] == "ORTA_KUSUR")
        kucuk_sayisi = sum(1 for k in siniflandirilmis_kusurlar if k["siddet"] == "KUCUK_KUSUR")

        toplam_kusurlu_alan = sum(k["alan"] for k in siniflandirilmis_kusurlar)

        if toplam_kusur == 0:
            parti_karari = "1_KALITE_PREMIUM"
            onay = True
        elif kritik_sayisi > 0:
            parti_karari = "PARTI_RED_HURDA"
            onay = False
        elif orta_sayisi > 0:
            parti_karari = "2_KALITE_SEVK"
            onay = True
        else:
            parti_karari = "REWORK_SONRASI_1_KALITE"
            onay = True

        return {
            "toplam_kusur_sayisi": toplam_kusur,
            "kritik_kusur_sayisi": kritik_sayisi,
            "orta_kusur_sayisi": orta_sayisi,
            "kucuk_kusur_sayisi": kucuk_sayisi,
            "toplam_kusurlu_alan_piksel": toplam_kusurlu_alan,
            "parti_kalite_karari": parti_karari,
            "parti_onayi": onay
        }
