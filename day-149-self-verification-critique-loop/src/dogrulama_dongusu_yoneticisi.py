"""
Doğrulama ve Eleştiri Döngüsü Yöneticisi Modülü (Day 149 - Faz 8).
Actor-Critic mimarisinde kendi kendine doğrulama ve rafine etme orkestratörü.
"""

from typing import Dict, Any, List
from .aktor_cozucu import AktorCozucu
from .elestirmen_dogrulayici import ElestirmenDogrulayici


class DogrulamaDongusuYoneticisi:
    """Actor-Critic döngüsünü yöneten ve çözümü ters sağlama ile kesinleştiren motor."""

    def __init__(self, maks_dongu: int = 3):
        self.maks_dongu = maks_dongu
        self.aktor = AktorCozucu()
        self.elestirmen = ElestirmenDogrulayici()

    def calistir(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actor-Critic döngüsünü çalıştırır:
        1. Üret (Generate) -> 2. Ters Sağlama Yap (Verify) -> 3. Eleştir (Critique) -> 4. Rafine Et (Refine)
        """
        dongu_kayitlari: List[Dict[str, Any]] = []

        # 1. İlk Çözüm Üretimi
        cozum = self.aktor.ilk_cozumu_uret(problem)

        for tur in range(1, self.maks_dongu + 1):
            # 2. Ters Sağlama (Reverse Verification)
            elestiri = self.elestirmen.ters_saglama_yap(problem, cozum)

            kayit = {
                "tur": tur,
                "aday_x": cozum["aday_x"],
                "dusunce_zinciri": cozum["dusunce_zinciri"],
                "guven_skoru": cozum["guven_skoru"],
                "dogrulandi_mi": elestiri["dogrulandi_mi"],
                "elestiri_notu": elestiri["elestiri_notu"],
                "hata_mesaji": elestiri["hata_mesaji"],
            }
            dongu_kayitlari.append(kayit)

            if elestiri["dogrulandi_mi"]:
                # Başarıyla doğrulandı, döngüden çık
                break

            # 3. Eleştiriye Göre Rafine Et (Refine)
            cozum = self.aktor.elestiriye_gore_rafine_et(problem, cozum, elestiri)

        return {
            "problem": problem,
            "toplam_tur_sayisi": len(dongu_kayitlari),
            "basarili_dogrulandi_mi": dongu_kayitlari[-1]["dogrulandi_mi"],
            "nihai_cozum": dongu_kayitlari[-1]["aday_x"],
            "nihai_zincir": dongu_kayitlari[-1]["dusunce_zinciri"],
            "dongu_kayitlari": dongu_kayitlari,
        }
