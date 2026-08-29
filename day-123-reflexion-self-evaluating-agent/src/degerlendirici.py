"""
Reflexion Değerlendirici (Evaluator) Modülü (Day 123 - Faz 7).
Üretilen kod veya yanıtları güvenlikli ortamda birim testlere tabi tutar ve skaler ödül (Reward) üretir.
"""

from typing import Dict, Any, List, Callable, Tuple
import traceback


class TestDurumu:
    """Tek bir birim test girdisi ve beklenen çıktısı."""
    __test__ = False

    def __init__(self, girdi: Any, beklenen: Any, aciklama: str = ""):
        self.girdi = girdi
        self.beklenen = beklenen
        self.aciklama = aciklama


BirimTestDurumu = TestDurumu


class KodDegerlendirici:
    """Üretilen fonksiyonları birim test kümesiyle değerlendiren modül."""

    def __init__(self, guvenli_builtins: Dict[str, Any] = None):
        self.builtins = guvenli_builtins or {
            "sum": sum, "len": len, "min": min, "max": max, "range": range,
            "abs": abs, "round": round, "int": int, "float": float, "str": str,
            "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
            "enumerate": enumerate, "sorted": sorted, "reversed": reversed,
        }

    def degerlendir(
        self,
        kod: str,
        fonksiyon_adi: str,
        test_kumesi: List[TestDurumu],
    ) -> Dict[str, Any]:
        """Kodu yürütür ve birim test başarı oranını hesaplar."""
        yerel_alan: Dict[str, Any] = {}

        # 1. Kod Derleme ve Tanımlama
        try:
            exec(kod, {"__builtins__": self.builtins}, yerel_alan)
        except Exception as e:
            return {
                "basarili": False,
                "odul": 0.0,
                "gecen_test": 0,
                "toplam_test": len(test_kumesi),
                "hata_tipi": "Syntax/CompileError",
                "hata_mesaji": f"Kod yürütme/derleme hatası: {type(e).__name__}: {str(e)}",
            }

        if fonksiyon_adi not in yerel_alan:
            return {
                "basarili": False,
                "odul": 0.0,
                "gecen_test": 0,
                "toplam_test": len(test_kumesi),
                "hata_tipi": "FunctionNotFoundError",
                "hata_mesaji": f"'{fonksiyon_adi}' adında fonksiyon bulunamadı.",
            }

        fonk = yerel_alan[fonksiyon_adi]
        gecen_testler = 0
        ilk_hata_mesaji = None
        ilk_hata_tipi = None

        # 2. Birim Testleri Koşturma
        for test in test_kumesi:
            try:
                if isinstance(test.girdi, tuple):
                    gercek_sonuc = fonk(*test.girdi)
                else:
                    gercek_sonuc = fonk(test.girdi)

                if gercek_sonuc == test.beklenen:
                    gecen_testler += 1
                else:
                    if ilk_hata_mesaji is None:
                        ilk_hata_tipi = "AssertionMismatch"
                        ilk_hata_mesaji = (
                            f"Test Başarısız ({test.aciklama}): Girdi={test.girdi}, "
                            f"Beklenen={test.beklenen}, Gerçekleşen={gercek_sonuc}"
                        )
            except Exception as e:
                if ilk_hata_mesaji is None:
                    ilk_hata_tipi = type(e).__name__
                    ilk_hata_mesaji = f"Test Çalışma Hatası ({test.aciklama}): Girdi={test.girdi}, Hata: {str(e)}"

        toplam = len(test_kumesi)
        odul = float(gecen_testler / max(1, toplam))
        basarili = (gecen_testler == toplam)

        return {
            "basarili": basarili,
            "odul": odul,
            "gecen_test": gecen_testler,
            "toplam_test": toplam,
            "hata_tipi": ilk_hata_tipi if not basarili else None,
            "hata_mesaji": ilk_hata_mesaji if not basarili else "Tüm testler eksiksiz geçti!",
        }
