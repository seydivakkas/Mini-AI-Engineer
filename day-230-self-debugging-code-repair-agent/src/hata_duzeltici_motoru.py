"""
Kendi Hatasını Düzelten (Self-Debugging) Kod Ajan Motoru (Day 230 - FAZ 12).
Test Geri Bildirimi, Reflexion Açıklaması ve Yinelemeli Kod Onarımı (Chen et al., 2023).
"""

from typing import Dict, Any, List, Optional, Tuple, Callable


class TestCase:
    """Birim Test Senaryosu Veri Yapısı."""
    __test__ = False

    def __init__(self, girdi: Any, beklenen: Any):
        self.girdi = girdi
        self.beklenen = beklenen


class ExecutionFeedback:
    """Test Çalıştırma ve Hata Teşhis Raporu."""

    def __init__(
        self,
        basarili_mi: bool,
        hata_turu: Optional[str] = None,
        hata_mesaji: Optional[str] = None,
        basarisiz_test: Optional[TestCase] = None,
        alinan_cikti: Optional[Any] = None,
    ):
        self.basarili_mi = basarili_mi
        self.hata_turu = hata_turu
        self.hata_mesaji = hata_mesaji
        self.basarisiz_test = basarisiz_test
        self.alinan_cikti = alinan_cikti


class CodeExecutionHarness:
    """Aday Kodları Birim Testlerle Denetleyen Test Çalıştırıcı."""

    @classmethod
    def testleri_kostur(
        cls,
        kod_metni: str,
        fonksiyon_adi: str,
        test_senaryolari: List[TestCase],
    ) -> ExecutionFeedback:
        """Kodu izole derler ve tüm testleri koşturur."""
        yerel_alan = {}
        try:
            exec(kod_metni, yerel_alan)
        except Exception as e:
            return ExecutionFeedback(
                basarili_mi=False,
                hata_turu="SyntaxError/CompileError",
                hata_mesaji=str(e),
            )

        if fonksiyon_adi not in yerel_alan:
            return ExecutionFeedback(
                basarili_mi=False,
                hata_turu="NameError",
                hata_mesaji=f"'{fonksiyon_adi}' adında bir fonksiyon tanımlanmadı.",
            )

        hedef_fonk = yerel_alan[fonksiyon_adi]

        for tc in test_senaryolari:
            try:
                if isinstance(tc.girdi, tuple):
                    cikti = hedef_fonk(*tc.girdi)
                else:
                    cikti = hedef_fonk(tc.girdi)

                if cikti != tc.beklenen:
                    return ExecutionFeedback(
                        basarili_mi=False,
                        hata_turu="AssertionError",
                        hata_mesaji=f"Girdi: {tc.girdi}, Beklenen: {tc.beklenen}, Alınan: {cikti}",
                        basarisiz_test=tc,
                        alinan_cikti=cikti,
                    )
            except Exception as e:
                return ExecutionFeedback(
                    basarili_mi=False,
                    hata_turu=type(e).__name__,
                    hata_mesaji=str(e),
                    basarisiz_test=tc,
                )

        return ExecutionFeedback(basarili_mi=True)


class SelfDebuggingAgent:
    """Reflexion Destekli Otonom Hata Düzeltici Ajan."""

    def __init__(self, max_deneme: int = 3):
        self.max_deneme = max_deneme
        self.debug_gunlugu: List[str] = []

    def onar_ve_coz(
        self,
        hedef_gorev: str,
        fonksiyon_adi: str,
        aday_kod_adimlari: List[Tuple[str, str]],
        test_senaryolari: List[TestCase],
    ) -> Dict[str, Any]:
        """
        aday_kod_adimlari: [(Kod, Reflexion_Aciklamasi)] dizilimi.
        """
        nihai_kod = None
        basarili = False
        toplam_adim = 0

        for adim, (kod, aciklama) in enumerate(aday_kod_adimlari, start=1):
            if adim > self.max_deneme:
                break

            toplam_adim = adim
            self.debug_gunlugu.append(f"--- [DENEME {adim}] Kod Derleniyor ve Test Ediliyor ---")

            geribildirim = CodeExecutionHarness.testleri_kostur(
                kod_metni=kod,
                fonksiyon_adi=fonksiyon_adi,
                test_senaryolari=test_senaryolari,
            )

            if geribildirim.basarili_mi:
                self.debug_gunlugu.append(f"✓ DENEME {adim} BAŞARILI! Tüm {len(test_senaryolari)} test eksiksiz geçti.")
                nihai_kod = kod
                basarili = True
                break
            else:
                self.debug_gunlugu.append(
                    f"❌ DENEME {adim} BAŞARISIZ ({geribildirim.hata_turu}): {geribildirim.hata_mesaji}\n"
                    f"💡 REFLEXION / DÜŞÜNÜM: {aciklama}"
                )

        return {
            "hedef_gorev": hedef_gorev,
            "basarili_mi": basarili,
            "toplam_deneme_sayisi": toplam_adim,
            "nihai_kod": nihai_kod,
            "debug_gunlugu": self.debug_gunlugu,
        }
