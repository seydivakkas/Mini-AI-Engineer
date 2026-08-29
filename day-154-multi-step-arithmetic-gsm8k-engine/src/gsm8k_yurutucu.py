"""
GSM8K İzole Yürütücü ve Değerlendirici Modülü (Day 154 - Faz 8).
PAL Python kodlarını güvenli yerel isim alanında çalıştırır ve ara adımları kaydeder.
"""

from typing import Dict, Any, List, Optional
import time


class GSM8KYurutucu:
    """Üretilen PAL kodlarını çalıştıran ve doğruluğunu ölçen motor."""

    @classmethod
    def kodu_calistir(cls, python_kodu: str) -> Dict[str, Any]:
        """
        Python kodunu izole isim alanında (local scope) çalıştırır ve 'solution()' sonucunu döner.
        """
        yerel_alan: Dict[str, Any] = {}
        baslangic = time.perf_counter()

        try:
            exec(python_kodu, {}, yerel_alan)
            if "solution" in yerel_alan and callable(yerel_alan["solution"]):
                sonuc = yerel_alan["solution"]()
                calisma_suresi = time.perf_counter() - baslangic
                return {
                    "basarili_mi": True,
                    "sonuc": sonuc,
                    "hata": None,
                    "calisma_suresi_ms": calisma_suresi * 1000,
                }
            else:
                return {
                    "basarili_mi": False,
                    "sonuc": None,
                    "hata": "'solution()' fonksiyonu bulunamadı.",
                    "calisma_suresi_ms": 0.0,
                }
        except Exception as e:
            return {
                "basarili_mi": False,
                "sonuc": None,
                "hata": str(e),
                "calisma_suresi_ms": 0.0,
            }

    @classmethod
    def cozum_karsilastir(
        cls,
        problem_adi: str,
        problem_metni: str,
        pal_kodu: str,
        beklenen_sonuc: float,
        raw_cot_tahmini: float,
    ) -> Dict[str, Any]:
        """
        Doğrudan LLM Aritmetiği (Raw CoT) ile PAL (Program-Aided Language Model) yaklaşımını kıyaslar.
        """
        pal_sonucu = cls.kodu_calistir(pal_kodu)
        pal_degeri = pal_sonucu["sonuc"]

        pal_dogru_mu = (pal_degeri == beklenen_sonuc)
        raw_cot_dogru_mu = (raw_cot_tahmini == beklenen_sonuc)

        return {
            "problem_adi": problem_adi,
            "problem_metni": problem_metni,
            "beklenen_sonuc": beklenen_sonuc,
            "pal_kodu": pal_kodu,
            "pal_sonucu": pal_degeri,
            "pal_dogru_mu": pal_dogru_mu,
            "raw_cot_tahmini": raw_cot_tahmini,
            "raw_cot_dogru_mu": raw_cot_dogru_mu,
            "calisma_suresi_ms": pal_sonucu["calisma_suresi_ms"],
        }
