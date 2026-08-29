"""
Reflexion Öz-Eleştiri (Self-Reflection / Reflector) Modülü (Day 123 - Faz 7).
Hatalı deneme çıktılarını ve değerlendirici hata mesajlarını sözel pekiştirmeli derse (Verbal Critique) dönüştürür.
"""

from typing import Dict, Any


class OzElestiriUreteci:
    """Başarısız kod denemelerini analiz edip bir sonraki deneme için sözel düzeltme üreten modül."""

    def elestiri_uret(
        self,
        deneme_no: int,
        hatali_kod: str,
        degerlendirme: Dict[str, Any],
    ) -> str:
        """Hata mesajını ve kodu inceleyerek somut bir öz-eleştiri cümlesi oluşturur."""
        hata_tipi = degerlendirme.get("hata_tipi", "BilinmeyenHata")
        hata_mesaji = degerlendirme.get("hata_mesaji", "")

        # 1. Mantık ve Sınır Durum (Edge Case) Eleştirisi
        if "AssertionMismatch" in hata_tipi:
            if "negatif" in hata_mesaji.lower() or "-" in hata_mesaji:
                return (
                    f"Öz-Eleştiri (Trial {deneme_no}): Kod negatif sayı içeren sınır durumlarında hatalı sonuç üretiyor. "
                    f"Başlangıç toplamı veya maksimum değer '0' yerine listenin ilk elemanı olarak başlatılmalı "
                    f"ve dinamik programlama / Kadane algoritması tam uygulanmalıdır."
                )
            else:
                return (
                    f"Öz-Eleştiri (Trial {deneme_no}): Beklenen çıktı ile gerçekleşen çıktı uyuşmuyor. "
                    f"Döngü sınırları ve koşul filtreleme mantığı gözden geçirilmeli, sınır durumları kapsanmalıdır."
                )

        # 2. İndeks ve Sınır Aşımı (IndexError) Eleştirisi
        elif "IndexError" in hata_tipi:
            return (
                f"Öz-Eleştiri (Trial {deneme_no}): Listede sınır dışı indeks erişimi (Off-by-one error) yapıldı. "
                f"Döngü 'range(len(dizi))' yerine 'len(dizi) - 1' sınırına göre ayarlanmalı veya boş dizi kontrolü eklenmelidir."
            )

        # 3. Tip ve Sözdizimi Hatası
        elif "TypeError" in hata_tipi or "Syntax/CompileError" in hata_tipi:
            return (
                f"Öz-Eleştiri (Trial {deneme_no}): Tip dönüşümü veya sözdizimsel hata tespit edildi ({hata_mesaji}). "
                f"Veri tiplerinin uyumluluğu doğrulanmalı ve Pythonic temiz yapılar kullanılmalıdır."
            )

        return (
            f"Öz-Eleştiri (Trial {deneme_no}): Testler başarısız oldu ({hata_mesaji}). "
            f"Algoritma mantığı baştan adım adım takip edilerek düzeltilmelidir."
        )
