"""
Doküman Metrikleri Değerlendirici Modülü (Day 165 - FAZ 9).
Levenshtein Distance, Normalized Edit Similarity (1 - NED) ve Karakter Doğruluğu hesaplayıcı.
"""

from typing import List, Dict, Any


class DokumanMetrikDegerlendirici:
    """OCR-Free Doküman Ayrıştırma için Metrik Motoru."""

    @classmethod
    def levenshtein_mesafesi(cls, s1: str, s2: str) -> int:
        """İki metin arasındaki minimum ekleme, silme, değiştirme işlem sayısı."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]

    @classmethod
    def normalized_edit_similarity(cls, tahmin: str, gercek: str) -> float:
        """
        Normalized Edit Distance (NED) tersi:
        Similarity = 1.0 - (Levenshtein(tahmin, gercek) / max(len(tahmin), len(gercek)))
        """
        max_len = max(len(tahmin), len(gercek))
        if max_len == 0:
            return 1.0

        dist = cls.levenshtein_mesafesi(tahmin, gercek)
        ned = dist / float(max_len)
        return max(0.0, float(1.0 - ned))

    @classmethod
    def toplu_degerlendir(cls, tahminler: List[str], gercekler: List[str]) -> Dict[str, Any]:
        """Tüm doküman çıktılarını NED ve Levenshtein ile değerlendirir."""
        skorlar = []
        for t, g in zip(tahminler, gercekler):
            skor = cls.normalized_edit_similarity(t, g)
            skorlar.append(skor)

        ortalama_sim = sum(skorlar) / len(skorlar) if skorlar else 0.0
        return {
            "toplam_dokuman": len(tahminler),
            "ortalama_edit_similarity": round(ortalama_sim, 4),
            "ortalama_dogruluk_yuzdesi": round(ortalama_sim * 100.0, 2),
            "bireysel_skorlar": [round(s, 4) for s in skorlar],
        }
