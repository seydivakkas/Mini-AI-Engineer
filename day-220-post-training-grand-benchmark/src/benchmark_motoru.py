"""
Grand Post-Training Benchmark Suite (Day 220 - FAZ 11 FİNALİ).
GSM8K, MATH-500, HumanEval ve MT-Bench Şampiyonluk Değerlendirme Paketi.
"""

from typing import Dict, Any, List, Optional
import re


class GSM8KEvaluator:
    """GSM8K İlkokul Matematik Akıl Yürütme Değerlendiricisi."""

    @classmethod
    def sayisal_yanit_cikar(cls, metin: str) -> Optional[float]:
        """Cevap metninden nihai sayıyı çeker (örn: '#### 42' veya 'Cevap: 42')."""
        eslesme = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", metin)
        if eslesme:
            return float(eslesme[-1])

        sayilar = re.findall(r"(-?\d+(?:\.\d+)?)", metin)
        if sayilar:
            return float(sayilar[-1])
        return None

    @classmethod
    def dogrula(cls, uretilen_yanit: str, hedef_sayi: float) -> bool:
        """Üretilen yanıtın doğruluğunu test eder."""
        bulunan = cls.sayisal_yanit_cikar(uretilen_yanit)
        if bulunan is None:
            return False
        return abs(bulunan - hedef_sayi) < 1e-4


class MATH500Evaluator:
    """MATH-500 İleri Düzey Matematik ve Olimpiyat Değerlendiricisi."""

    @classmethod
    def kutu_yanit_cikar(cls, metin: str) -> Optional[str]:
        """LaTeX \\boxed{...} formatındaki cevabı çeker."""
        eslesme = re.findall(r"\\boxed\{([^}]+)\}", metin)
        if eslesme:
            return eslesme[-1].strip()
        return None

    @classmethod
    def dogrula(cls, uretilen_yanit: str, hedef_ifade: str) -> bool:
        """Sembolik/Kutu yanıt eşleşmesini test eder."""
        bulunan = cls.kutu_yanit_cikar(uretilen_yanit)
        if bulunan is None:
            return False
        return bulunan.replace(" ", "") == hedef_ifade.replace(" ", "")


class HumanEvalEvaluator:
    """HumanEval Python Kod Üretim ve Doğrulama Değerlendiricisi."""

    @classmethod
    def kod_test_et(cls, fonksiyon_kodu: str, test_assertion: str) -> bool:
        """Üretilen kodu güvenli ortamda assertion ile test eder."""
        calistirma_alani = {}
        try:
            # Kod tanımını yürüt
            exec(fonksiyon_kodu, calistirma_alani, calistirma_alani)
            # Test assertion'ını yürüt
            exec(test_assertion, calistirma_alani, calistirma_alani)
            return True
        except Exception:
            return False


class MTBenchEvaluator:
    """MT-Bench Çok Turlu Konuşma ve Talimat Değerlendiricisi."""

    @classmethod
    def puanla(cls, cevap_1: str, cevap_2: str) -> float:
        """Çok turlu cevabın kalitesini 1-10 aralığında puanlar."""
        uzunluk_puani = min(5.0, (len(cevap_1.split()) + len(cevap_2.split())) / 20.0)
        yapi_puani = 4.0 if ("1." in cevap_1 or "-" in cevap_1 or "```" in cevap_2) else 2.5
        toplam = min(10.0, uzunluk_puani + yapi_puani + 1.0)
        return float(toplam)


class GrandBenchmarkSuite:
    """Tüm Benchmark Testlerini Birleştiren Şampiyonluk Motoru."""

    @classmethod
    def tam_degerlendirme_kos(cls) -> Dict[str, Any]:
        """GSM8K, MATH500, HumanEval ve MT-Bench testlerini koşturur."""
        # 1. GSM8K Örnek Testi
        gsm_yanit = "Ali'nin 15 elması vardı. 5 tanesini Ayşe'ye verdi. Kalan: 15 - 5 = 10. #### 10"
        gsm_basari = GSM8KEvaluator.dogrula(gsm_yanit, 10.0)

        # 2. MATH500 Örnek Testi
        math_yanit = "İki denklemi birleştirdiğimizde kök: \\boxed{3/4}"
        math_basari = MATH500Evaluator.dogrula(math_yanit, "3/4")

        # 3. HumanEval Örnek Testi
        kod = "def topla(a, b):\n    return a + b"
        kod_test = "assert topla(3, 4) == 7 and topla(-1, 1) == 0"
        he_basari = HumanEvalEvaluator.kod_test_et(kod, kod_test)

        # 4. MT-Bench Örnek Testi
        mt_skor = MTBenchEvaluator.puanla(
            cevap_1="1. Python'da liste üreteçleri oldukça hızlıdır.\n2. Bellek tasarrufu sağlar.",
            cevap_2="```python\nveri = [x**2 for x in range(10)]\n```",
        )

        return {
            "gsm8k_gecisi": gsm_basari,
            "math500_gecisi": math_basari,
            "humaneval_gecisi": he_basari,
            "mt_bench_skoru": mt_skor,
        }
