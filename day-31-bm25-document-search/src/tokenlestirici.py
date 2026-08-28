"""
Metin Ön İşleme ve Tokenizasyon Motoru (Text Tokenizer & Normalizer).
"""

from typing import List, Set
import re
import unicodedata


class MetinTokenlestirici:
    """
    Ham metinleri temizleyen, normalize eden ve leksikal analiz için
    kelime jetonlarına (tokens) ayrıştıran sınıf.
    """

    DEFAULT_STOP_WORDS: Set[str] = {
        "ve", "ile", "veya", "bu", "şu", "o", "bir", "de", "da", "ki", "için", "olan",
        "olarak", "gibi", "ile", "en", "daha", "çok", "ne", "mi", "mu", "mü", "mı",
        "the", "a", "an", "and", "or", "in", "on", "at", "of", "to", "for", "with", "is", "are"
    }

    def __init__(self, stop_words: Set[str] = None, min_token_len: int = 2):
        self.stop_words = set(stop_words) if stop_words is not None else self.DEFAULT_STOP_WORDS
        self.min_token_len = min_token_len

    def normalize_et(self, metin: str) -> str:
        """Küçük harfe dönüştürür ve noktalama işaretlerini ayıklar."""
        metin = metin.strip().lower()
        # Harf ve rakam haricindeki karakterleri boşluğa dönüştür
        metin = re.sub(r"[^\w\s]", " ", metin, flags=re.UNICODE)
        # Çoklu boşlukları teke indir
        metin = re.sub(r"\s+", " ", metin)
        return metin

    def tokenlestir(self, metin: str, stop_words_filtrele: bool = True) -> List[str]:
        """Metni kelime tokenlarına ayırır."""
        temiz_metin = self.normalize_et(metin)
        ham_tokenlar = temiz_metin.split()

        tokenlar = []
        for t in ham_tokenlar:
            if len(t) < self.min_token_len:
                continue
            if stop_words_filtrele and t in self.stop_words:
                continue
            tokenlar.append(t)

        return tokenlar
