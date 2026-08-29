"""
Büyük Öğretmen Model Akıl Yürütme Simülatörü Modülü (Day 158 - Faz 8).
DeepSeek-R1 (671B) tarzı büyük modelin ürettiği ham düşünce izlerini (<think>...</think>) simüle eder.
"""

from typing import Dict, Any, List


class OgretmenModelSimulasyonu:
    """Büyük akıl yürüten öğretmenin düşünce izi üreticisi."""

    @classmethod
    def iz_uret(cls, soru: str, senaryo: str = "mukemmel") -> Dict[str, Any]:
        """
        Soru için öğretmen modelin ürettiği düşünce zincirini ve cevabı döner.
        """
        if senaryo == "mukemmel":
            dusunce = (
                "<think>\n"
                "Kullanıcı bir cebir problemi soruyor: 3x + 15 = 45 denkleminin kökünü bulunuz.\n"
                "Adım 1: Denklemin her iki tarafından 15 çıkaralım: 3x = 45 - 15 = 30.\n"
                "Adım 2: Şimdi her iki tarafı 3'e bölelim: x = 30 / 3 = 10.\n"
                "Dur bir dakika, sağlama yapalım (Self-Verification): 3*(10) + 15 = 30 + 15 = 45. Doğru!\n"
                "Her şey tutarlı, nihai cevabı formatlayabilirim.\n"
                "</think>\n"
                "Verilen denklemin çözümü: x = 10'dur."
            )
            dogru_mu = True
            token_sayisi = 120
            refleksif_ifade_sayisi = 2

        elif senaryo == "dongulu_hatali":
            dusunce = (
                "<think>\n"
                "x'i hesaplayalım. 3x + 15 = 45. Acaba x=5 mi? 3*5+15=30 olmadı. "
                "Acaba x=6 mı? 3*6+15=33 olmadı. Tekrar dene... " * 15 +
                "</think>\n"
                "Sonuç bulunamadı."
            )
            dogru_mu = False
            token_sayisi = 450
            refleksif_ifade_sayisi = 0

        else: # hatali_sonuc
            dusunce = (
                "<think>\n"
                "3x + 15 = 45 ise 3x = 60 ve x = 20.\n"
                "</think>\n"
                "Cevap: x = 20."
            )
            dogru_mu = False
            token_sayisi = 45
            refleksif_ifade_sayisi = 0

        return {
            "soru": soru,
            "ham_iz": dusunce,
            "dogru_mu": dogru_mu,
            "token_sayisi": token_sayisi,
            "refleksif_ifade_sayisi": refleksif_ifade_sayisi,
            "senaryo": senaryo,
        }
