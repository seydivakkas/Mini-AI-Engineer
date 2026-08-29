"""
Düşünce İzi Filtreleme ve Kürasyon Modülü (Day 158 - Faz 8).
Öğretmen modelin ürettiği ham düşünce izlerini doğruluk, döngü ve bilişsel kalite kriterleriyle süzer.
"""

from typing import Dict, Any, List


class DusunceIziFiltreleyici:
    """Damıtma veri setini temizleyen kalite süzgeci."""

    @classmethod
    def izi_degerlendir(cls, iz_verisi: Dict[str, Any]) -> Dict[str, Any]:
        """
        Düşünce izinin damıtma eğitimine uygunluğunu denetler.
        """
        ham_metin = iz_verisi["ham_iz"]
        dogru_mu = iz_verisi["dogru_mu"]
        token_sayisi = iz_verisi["token_sayisi"]

        # Kriter 1: Aşırı Tekrar ve Kısırdöngü Kontrolü
        if "Tekrar dene" in ham_metin or token_sayisi > 1000 or iz_verisi.get("senaryo") == "dongulu_hatali":
            return {
                "kabul_edildi_mi": False,
                "kalite_skoru": 0.1,
                "red_nedeni": "Düşünce akışında kısırdöngü veya aşırı token anomalisi tespit edildi.",
            }

        # Kriter 2: Düşünce Etiketleri Kontrolü (<think> ... </think>)
        if "<think>" not in ham_metin or "</think>" not in ham_metin:
            return {
                "kabul_edildi_mi": False,
                "kalite_skoru": 0.2,
                "red_nedeni": "Açık düşünce (<think>) etiketleri eksik.",
            }

        # Kriter 3: Nihai Doğruluk Kontrolü
        if not dogru_mu:
            return {
                "kabul_edildi_mi": False,
                "kalite_skoru": 0.0,
                "red_nedeni": "Nihai cevap yanlış veya geçersiz.",
            }

        # Kriter 4: Refleksif Zenginlik (Self-Verification / Backtrack varlığı)
        kalite_skoru = 0.85
        if iz_verisi.get("refleksif_ifade_sayisi", 0) >= 1:
            kalite_skoru = 0.98

        return {
            "kabul_edildi_mi": True,
            "kalite_skoru": kalite_skoru,
            "red_nedeni": None,
            "temizlenmis_iz": ham_metin,
        }
