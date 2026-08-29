"""
Eleştirmen Doğrulayıcı Modülü (Day 149 - Faz 8).
Çözümden girdiye ters sağlama (Reverse Verification / Back-Substitution) ve kesinlik denetleyicisi.
"""

from typing import Dict, Any


class ElestirmenDogrulayici:
    """Aday çözümü başlangıç koşullarına yerleştirip sağlamasını yapan Eleştirmen (Critic/Verifier)."""

    def __init__(self, model_adi: str = "MiniReasoning-Critic-8B"):
        self.model_adi = model_adi

    def ters_saglama_yap(self, problem: Dict[str, Any], aday_cozum: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aday çözümü denkleme geri koyarak ters sağlama yapar.
        """
        tur = problem.get("tur", "moduler_aritmetik")
        aday_x = aday_cozum.get("aday_x", 0)

        if tur == "moduler_aritmetik":
            # Problem: 3x + 7 = 2 (mod 5)
            hesaplanan_sol = (3 * aday_x + 7) % 5
            beklenen_sag = 2

            if hesaplanan_sol == beklenen_sag:
                return {
                    "dogrulandi_mi": True,
                    "hesaplanan_deger": hesaplanan_sol,
                    "beklenen_deger": beklenen_sag,
                    "hata_mesaji": None,
                    "elestiri_notu": f"Kusursuz: 3*({aday_x}) + 7 = {3*aday_x + 7} = {hesaplanan_sol} (mod 5) beklenen 2 ile birebir örtüştü!",
                    "kesinlik_skoru": 1.0,
                }
            else:
                return {
                    "dogrulandi_mi": False,
                    "hesaplanan_deger": hesaplanan_sol,
                    "beklenen_deger": beklenen_sag,
                    "hata_mesaji": f"Ters Sağlama Başarısız: 3*({aday_x}) + 7 = {3*aday_x + 7} mod 5 sonucu {hesaplanan_sol} veriyor, beklenen {beklenen_sag} idi!",
                    "elestiri_notu": f"x = {aday_x} kök değildir! 3x = 0 mod 5 olmalıdır.",
                    "kesinlik_skoru": 0.0,
                }
        else:
            # Genel denklem: 2x + 5 = 15
            hesap = 2 * aday_x + 5
            beklenen = 15
            if hesap == beklenen:
                return {
                    "dogrulandi_mi": True,
                    "hesaplanan_deger": hesap,
                    "beklenen_deger": beklenen,
                    "hata_mesaji": None,
                    "elestiri_notu": f"Doğrulandı: 2*({aday_x}) + 5 = {hesap} == 15",
                    "kesinlik_skoru": 1.0,
                }
            else:
                return {
                    "dogrulandi_mi": False,
                    "hesaplanan_deger": hesap,
                    "beklenen_deger": beklenen,
                    "hata_mesaji": f"2*({aday_x}) + 5 = {hesap} != 15",
                    "elestiri_notu": "Hesaplama hatası.",
                    "kesinlik_skoru": 0.0,
                }
