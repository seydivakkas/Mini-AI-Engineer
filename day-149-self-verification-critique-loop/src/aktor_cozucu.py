"""
Aktör Çözücü Modülü (Day 149 - Faz 8).
Problem çözümü için ilk akıl yürütme zincirini ve eleştiriye dayalı düzeltmeleri üreten Aktör (Generator).
"""

from typing import Dict, Any, List


class AktorCozucu:
    """Problem çözümü ve revizyon üreten Aktör modeli simülasyonu."""

    def __init__(self, model_adi: str = "MiniReasoning-Actor-8B"):
        self.model_adi = model_adi

    def ilk_cozumu_uret(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Verilen problem için ilk taslak çözümü üretir."""
        tur = problem.get("tur", "moduler_aritmetik")

        if tur == "moduler_aritmetik":
            # Problem: 3x + 7 = 2 (mod 5)
            # İlk taslakta aceleci/hatalı bir tahmin simüle edilir
            return {
                "aday_x": 2,
                "dusunce_zinciri": [
                    "1. Denklem: 3x + 7 = 2 (mod 5)",
                    "2. 7 = 2 (mod 5) olduğu için 3x + 2 = 2 (mod 5)",
                    "3. 3x = 0 (mod 5), tahminimce x = 2", # Hatalı adım
                ],
                "guven_skoru": 0.65,
                "tur_sayisi": 1,
            }
        else:
            # Genel denklem: 2x + 5 = 15
            return {
                "aday_x": 6, # Hatalı taslak
                "dusunce_zinciri": ["2x + 5 = 15 => 2x = 12 => x = 6"],
                "guven_skoru": 0.60,
                "tur_sayisi": 1,
            }

    def elestiriye_gore_rafine_et(self, problem: Dict[str, Any], onceki_cozum: Dict[str, Any], elestiri_raporu: Dict[str, Any]) -> Dict[str, Any]:
        """Eleştirmenin tespit ettiği hataya göre çözümü yeniden hesaplar."""
        tur = problem.get("tur", "moduler_aritmetik")

        if tur == "moduler_aritmetik":
            # Düzeltilmiş doğru akıl yürütme
            return {
                "aday_x": 0,
                "dusunce_zinciri": [
                    f"<think> Eleştirmen uyardı: {elestiri_raporu['hata_mesaji']}. Yeniden hesaplıyorum. </think>",
                    "1. 3x + 7 = 2 (mod 5)",
                    "2. 7 mod 5 = 2 => 3x + 2 = 2 (mod 5)",
                    "3. Her iki taraftan 2 çıkarılır: 3x = 0 (mod 5)",
                    "4. gcd(3, 5) = 1 olduğu için 3'ün mod 5'te tersi vardır (2*3=6=1)",
                    "5. x = 0 * 2 = 0 (mod 5). O halde x = 0!",
                ],
                "guven_skoru": 0.99,
                "tur_sayisi": onceki_cozum["tur_sayisi"] + 1,
            }
        else:
            return {
                "aday_x": 5,
                "dusunce_zinciri": ["2x = 10 => x = 5"],
                "guven_skoru": 0.99,
                "tur_sayisi": onceki_cozum["tur_sayisi"] + 1,
            }
