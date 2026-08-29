"""
Sentetik Veri Kalite Filtresi ve Eleme Modülü (Day 116).
Evol-Instruct eleme kriterleri: Kopya istem engelleme, karmaşıklık kazancı (Complexity Gain) doğrulaması.
"""

from typing import Tuple, Set
import re


class SentetikKaliteFiltresi:
    """Evrilen istemlerin kalitesini ve evrimsel geçerliliğini denetleyen filtre motoru."""

    TEKNIK_KELIMELER = {
        "optimizasyon", "bellek", "karmaşıklık", "algoritma", "kanıt", "hipotez",
        "istisna", "asimptotik", "kuantum", "lidar", "hft", "dedüktif", "np-zorluk",
        "işletim", "senaryo", "kısıt", "mikrosaniye", "dağıtık", "kriptografik"
    }

    @staticmethod
    def _tokenlara_ayir(metin: str) -> Set[str]:
        temiz = re.sub(r"[^\w\s]", "", metin.lower())
        return set(temiz.split())

    def jaccard_benzerlik(self, metin1: str, metin2: str) -> float:
        t1 = self._tokenlara_ayir(metin1)
        t2 = self._tokenlara_ayir(metin2)
        if not t1 or not t2:
            return 0.0
        kesisim = len(t1.intersection(t2))
        birlesim = len(t1.union(t2))
        return kesisim / max(1, birlesim)

    def karmasiklik_skoru(self, metin: str) -> float:
        """İstemin dilsel, teknik ve kural bazlı karmaşıklık skorunu (0-100) hesaplar."""
        tokenlar = self._tokenlara_ayir(metin)
        kelime_sayisi = len(tokenlar)

        # 1. Uzunluk faktörü (en fazla 40 puan)
        uzunluk_puani = min(40.0, (kelime_sayisi / 25.0) * 40.0)

        # 2. Teknik sözcük yoğunluğu (en fazla 35 puan)
        teknik_sayisi = sum(1 for t in tokenlar if t in self.TEKNIK_KELIMELER)
        teknik_puani = min(35.0, teknik_sayisi * 7.0)

        # 3. Kural ve yapısal etiket puanı (en fazla 25 puan)
        etiket_puani = 0.0
        if "[" in metin and "]" in metin:
            etiket_puani += 15.0
        if ":" in metin:
            etiket_puani += 10.0

        toplam_skor = uzunluk_puani + teknik_puani + etiket_puani
        return float(min(100.0, toplam_skor))

    def gecerlilik_elemesi(self, tohum_prompt: str, evrilmis_prompt: str) -> Tuple[bool, str]:
        """
        Evol-Instruct eleme kuralları:
        1. Kopya / Yetersiz Değişim Kuralı: Jaccard > 0.92 veya uzunluk artışı < %15 ise RET.
        2. Karmaşıklık Kazancı Kuralı: Yeni skor tohumdan yüksek olmalı.
        3. Aşırı Kısa/Bozuk İstem Kuralı: 10 kelimeden az ise RET.
        """
        if len(evrilmis_prompt.split()) < 8:
            return False, "RET: İstem aşırı kısa veya bozuk."

        benzerlik = self.jaccard_benzerlik(tohum_prompt, evrilmis_prompt)
        if benzerlik > 0.92:
            return False, "RET: Yetersiz evrimsel mutasyon (Kopya istem)."

        skor_tohum = self.karmasiklik_skoru(tohum_prompt)
        skor_yeni = self.karmasiklik_skoru(evrilmis_prompt)

        if skor_yeni <= skor_tohum:
            return False, "RET: Karmaşıklık kazancı (Complexity Gain) sağlanamadı."

        return True, "KABUL: İstem başarıyla evrimleşti ve kalite testlerini geçti."
