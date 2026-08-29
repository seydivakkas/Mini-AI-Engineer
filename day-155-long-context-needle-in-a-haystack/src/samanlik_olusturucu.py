"""
Samanlık ve İğne Oluşturucu Modülü (Day 155 - Faz 8).
Belirtilen uzunlukta sentetik doküman (Haystack) üretir ve hedeflenen derinlik yüzdesinde iğneyi (Needle) enjekte eder.
"""

from typing import Dict, Any, List


class SamanlikOlusturucu:
    """Uzun bağlamlı samanlık metni oluşturan ve iğne enjekte eden motor."""

    DOLGU_CUMLESI = "Yapay zeka modelleri büyük veri kümeleri üzerinde eğitilerek dil ve görsel kalıplarını öğrenir. "

    @classmethod
    def samanlik_uret(
        cls,
        hedef_kelime_sayisi: int,
        igne_metni: str,
        derinlik_yuzdesi: float, # 0.0 (%0) ile 1.0 (%100) arası
    ) -> Dict[str, Any]:
        """
        İstenen kelime uzunluğunda doküman üretir ve iğneyi belirtilen derinliğe yerleştirir.
        """
        dolgu_kelime_sayisi = len(cls.DOLGU_CUMLESI.split())
        gerekli_tekrar = max(1, hedef_kelime_sayisi // dolgu_kelime_sayisi)

        paragraflar = [cls.DOLGU_CUMLESI.strip() for _ in range(gerekli_tekrar)]

        # İğnenin yerleştirileceği indeks
        derinlik = max(0.0, min(1.0, derinlik_yuzdesi))
        ekleme_indeksi = int(len(paragraflar) * derinlik)

        # İğneyi enjekte et
        paragraflar.insert(ekleme_indeksi, f"--- KRİTİK BİLGİ: {igne_metni} ---")

        tam_dokuman = " ".join(paragraflar)
        toplam_kelime = len(tam_dokuman.split())

        return {
            "tam_dokuman": tam_dokuman,
            "igne_metni": igne_metni,
            "derinlik_yuzdesi": derinlik * 100.0,
            "ekleme_indeksi": ekleme_indeksi,
            "toplam_kelime_sayisi": toplam_kelime,
            "toplam_paragraf_sayisi": len(paragraflar),
        }
