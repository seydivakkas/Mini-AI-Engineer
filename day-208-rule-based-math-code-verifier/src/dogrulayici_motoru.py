"""
Kural Tabanlı Doğrulayıcı (Rule-Based Verifiers) Motoru (Day 208 - FAZ 11).
SymPy Sembolik Matematik, Python AST Sentaks Denetimi ve Güvenli Birim Test Yürütücüsü.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import ast
import re
import sympy as sp


class SymPyMathVerifier:
    """
    SymPy Tabanlı Deterministik Sembolik Matematik Doğrulayıcısı.
    Karakter eşleşmesi yerine cebirsel denkliği (Algebraic Equivalence) doğrular.
    """

    @classmethod
    def _ifadeyi_temizle(cls, metin: str) -> str:
        """LaTeX ve ham metin matematik sembollerini SymPy formatına çevirir."""
        temiz = metin.strip()
        # 1. \sqrt{a} -> sqrt(a) (Önce iç kökleri temizle ki \frac içindeki parantezler düzelsin)
        temiz = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", temiz)
        # 2. \frac{a}{b} -> (a)/(b)
        temiz = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", temiz)
        # 3. ^ -> **
        temiz = temiz.replace("^", "**")
        # 4. Gereksiz LaTeX etiketleri
        temiz = temiz.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*")
        return temiz

    @classmethod
    def sembolik_esitlik_kontrolu(cls, ifade_aday: str, ifade_hedef: str) -> bool:
        """
        İki matematiksel ifadenin sembolik olarak özdeş olup olmadığını doğrular:
        simplify(expr_aday - expr_hedef) == 0
        """
        try:
            aday_str = cls._ifadeyi_temizle(ifade_aday)
            hedef_str = cls._ifadeyi_temizle(ifade_hedef)

            expr_a = sp.sympify(aday_str)
            expr_h = sp.sympify(hedef_str)

            fark = sp.simplify(expr_a - expr_h)
            if fark == 0:
                return True

            # Sayısal yaklaşık eşitlik kontrolü (Float toleransı)
            val_a = float(expr_a.evalf())
            val_h = float(expr_h.evalf())
            return abs(val_a - val_h) < 1e-6
        except Exception:
            return False

    @classmethod
    def denklem_cozumu_dogrula(
        cls,
        denklem_str: str,
        aday_cozum: Union[str, float, int],
        degisken: str = "x",
    ) -> bool:
        """Verilen adayın denklemi (LHS == RHS) sağlayıp sağlamadığını doğrular."""
        try:
            sol, sag = denklem_str.split("=")
            var = sp.Symbol(degisken)

            expr_sol = sp.sympify(cls._ifadeyi_temizle(sol))
            expr_sag = sp.sympify(cls._ifadeyi_temizle(sag))
            val_cozum = sp.sympify(cls._ifadeyi_temizle(str(aday_cozum)))

            sol_sonuc = expr_sol.subs(var, val_cozum)
            sag_sonuc = expr_sag.subs(var, val_cozum)

            return sp.simplify(sol_sonuc - sag_sonuc) == 0
        except Exception:
            return False


class PythonASTCodeVerifier:
    """
    Python AST (Abstract Syntax Tree) Statik Kod Doğrulayıcısı.
    Çalıştırmadan önce sentaks geçerliliğini ve güvenlik kısıtlarını denetler.
    """

    YASAKLI_FONKSIYONLAR = {"eval", "exec", "open", "__import__", "compile"}
    YASAKLI_MODULLER = {"os", "sys", "subprocess", "shutil", "socket", "requests", "urllib"}

    @classmethod
    def sentaks_dogrula(cls, kod_metni: str) -> Tuple[bool, Optional[str]]:
        """Kodun Python sentaksına uygunluğunu parse eder."""
        try:
            ast.parse(kod_metni)
            return True, None
        except SyntaxError as e:
            return False, f"Sentaks Hatası (Satır {e.lineno}): {e.msg}"

    @classmethod
    def guvenlik_ve_yapi_denetle(cls, kod_metni: str) -> Tuple[bool, Optional[str]]:
        """Zararlı import ve fonksiyon çağrılarını AST düğümleriyle yakalar."""
        try:
            tree = ast.parse(kod_metni)
        except SyntaxError:
            return False, "Geçersiz sentaks."

        for node in ast.walk(tree):
            # Yasaklı import denetimi
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in cls.YASAKLI_MODULLER:
                        return False, f"Yasaklı modül import edildi: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module in cls.YASAKLI_MODULLER:
                    return False, f"Yasaklı modülden import yapıldı: {node.module}"
            # Yasaklı yerleşik fonksiyon çağrısı
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.YASAKLI_FONKSIYONLAR:
                    return False, f"Yasaklı fonksiyon çağrısı: {node.func.id}"

        return True, "Kod güvenlik kural setine uygundur."


class DeterministicUnitTestRunner:
    """
    İzole ve Deterministik Fonksiyon Birim Test Yürütücüsü.
    """

    @classmethod
    def birim_testleri_calistir(
        cls,
        fonksiyon_kodu: str,
        fonksiyon_adi: str,
        test_vakaları: List[Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        """
        test_vakaları: [(girdi_args, beklenen_cikti), ...]
        """
        # Sentaks ve güvenlik kontrolü
        gecerli, err = PythonASTCodeVerifier.guvenlik_ve_yapi_denetle(fonksiyon_kodu)
        if not gecerli:
            return {"basarili": False, "gecen_test": 0, "toplam_test": len(test_vakaları), "hata": err}

        yerel_alan = {}
        try:
            exec(fonksiyon_kodu, {}, yerel_alan)
            if fonksiyon_adi not in yerel_alan:
                return {
                    "basarili": False,
                    "gecen_test": 0,
                    "toplam_test": len(test_vakaları),
                    "hata": f"Fonksiyon bulunamadı: {fonksiyon_adi}",
                }

            hedef_fn = yerel_alan[fonksiyon_adi]
            gecen = 0

            for girdi, beklenen in test_vakaları:
                if isinstance(girdi, tuple):
                    sonuc = hedef_fn(*girdi)
                else:
                    sonuc = hedef_fn(girdi)

                if sonuc == beklenen:
                    gecen += 1

            basari_orani = gecen / len(test_vakaları) if test_vakaları else 0.0
            return {
                "basarili": gecen == len(test_vakaları),
                "gecen_test": gecen,
                "toplam_test": len(test_vakaları),
                "basari_orani": basari_orani,
                "hata": None,
            }
        except Exception as e:
            return {
                "basarili": False,
                "gecen_test": 0,
                "toplam_test": len(test_vakaları),
                "hata": f"Çalışma Zamanı Hatası: {str(e)}",
            }


class RuleBasedRewardEngine:
    """
    DeepSeek-R1 ve RLVR Tarzı Kural Tabanlı Ödül Motoru.
    Format + Sembolik Matematik / Kod Birim Testi birleşimi.
    """

    @classmethod
    def matematik_odulu_hesapla(
        cls,
        model_yaniti: str,
        hedef_ifade: str,
    ) -> Dict[str, float]:
        """Biçim ve SymPy sembolik eşitlik ödülünü hesaplar."""
        has_think = "<think>" in model_yaniti and "</think>" in model_yaniti
        format_odul = 0.20 if has_think else 0.0

        # </think> sonrası son cevabı ayıkla
        if has_think:
            icerik = model_yaniti.split("</think>")[-1]
        else:
            icerik = model_yaniti

        # \boxed{...} veya sayıyı ara
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", icerik)
        if boxed_match:
            aday_cevap = boxed_match.group(1)
        else:
            sayilar = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", icerik)
            aday_cevap = sayilar[-1] if sayilar else icerik.strip()

        dogru_mu = SymPyMathVerifier.sembolik_esitlik_kontrolu(aday_cevap, hedef_ifade)
        dogruluk_odul = 0.80 if dogru_mu else 0.0

        return {
            "format_odulu": format_odul,
            "dogruluk_odulu": dogruluk_odul,
            "toplam_odul": format_odul + dogruluk_odul,
            "aday_cevap": aday_cevap,
            "eslesme": dogru_mu,
        }
