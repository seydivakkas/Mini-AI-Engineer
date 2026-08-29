"""
Whisper Hata ve Zaman Damgası Değerlendirici Modülü (Day 170 - FAZ 9).
WER (Word Error Rate), CER (Character Error Rate) ve Levenshtein mesafesini hesaplar.
"""

from typing import List, Dict, Any


class WhisperMetrikDegerlendirici:
    """ASR Başarı ve Hata Metrikleri Motoru."""

    @classmethod
    def levenshtein(cls, a: List[str], b: List[str]) -> int:
        """İki dize veya kelime listesi arasındaki düzenleme mesafesi."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    @classmethod
    def wer_hesapla(cls, gercek: str, tahmin: str) -> float:
        """Kelime Hata Oranı (Word Error Rate - WER)."""
        g_kelimeler = gercek.strip().split()
        t_kelimeler = tahmin.strip().split()
        if not g_kelimeler:
            return 0.0 if not t_kelimeler else 1.0
        mesafe = cls.levenshtein(g_kelimeler, t_kelimeler)
        return round(float(mesafe / len(g_kelimeler)), 4)

    @classmethod
    def cer_hesapla(cls, gercek: str, tahmin: str) -> float:
        """Karakter Hata Oranı (Character Error Rate - CER)."""
        g_karakterler = list(gercek.strip())
        t_karakterler = list(tahmin.strip())
        if not g_karakterler:
            return 0.0 if not t_karakterler else 1.0
        mesafe = cls.levenshtein(g_karakterler, t_karakterler)
        return round(float(mesafe / len(g_karakterler)), 4)
