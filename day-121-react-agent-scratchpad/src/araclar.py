"""
ReAct Ajanı Araç Seti ve Kayıt Defteri Modülü (Day 121 - Faz 7).
Ajanın dış dünya ile etkileşime geçmesini sağlayan güvenli araçlar (Hesap Makinesi, Arama Motoru, Kod Yürütücü, SQL).
"""

from typing import Dict, Any, Callable
import ast
import operator
import re


class TemelArac:
    """Tüm ajan araçları için temel soyut sınıf."""

    def __init__(self, ad: str, aciklama: str):
        self.ad = ad
        self.aciklama = aciklama

    def calistir(self, girdi: str) -> str:
        raise NotImplementedError("Her araç 'calistir' metodunu uygulamalıdır.")


class HesapMakinasi(TemelArac):
    """AST tabanlı güvenli matematiksel hesaplama aracı."""

    ISLEMLER = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def __init__(self):
        super().__init__(
            ad="HesapMakinasi",
            aciklama="Matematiksel ifadeleri kesin olarak hesaplar. Girdi: Örn. '25 * 4 + 18'",
        )

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return self.ISLEMLER[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.ISLEMLER[type(node.op)](self._eval(node.operand))
        else:
            raise ValueError(f"Desteklenmeyen AST düğümü: {type(node)}")

    def calistir(self, girdi: str) -> str:
        try:
            # Sadece matematiksel karakterlere izin ver
            temiz = girdi.strip().replace(" ", "")
            tree = ast.parse(temiz, mode="eval")
            sonuc = self._eval(tree.body)
            return f"Hesaplama Sonucu: {sonuc}"
        except Exception as e:
            return f"Hesaplama Hatası: Geçersiz matematiksel ifade ({str(e)})"


class AramaMotoru(TemelArac):
    """Simüle edilmiş bilgi getirme ve arama motoru aracı."""

    BILGI_TABANI = {
        "türkiye başkenti": "Türkiye'nin başkenti Ankara'dır. Nüfusu yaklaşık 5.8 milyondur.",
        "ankara nüfus": "Ankara'nın 2024 yılı itibarıyla nüfusu 5,803,482 kişidir.",
        "istanbul nüfus": "İstanbul'un 2024 nüfusu 15,655,924 kişidir.",
        "python çıkış yılı": "Python programlama dili ilk olarak 1991 yılında Guido van Rossum tarafından yayınlanmıştır.",
        "deepseek-r1": "DeepSeek-R1, açık kaynaklı ve akıl yürütme (reasoning) yeteneklerine sahip 671B MoE LLM modelidir.",
        "react paper": "ReAct: Synergizing Reasoning and Acting in Language Models makalesi 2022 yılında Shunyu Yao ve ekibi tarafından yayınlanmıştır.",
    }

    def __init__(self):
        super().__init__(
            ad="AramaMotoru",
            aciklama="İnternet veya bilgi tabanında güncel bilgi arar. Girdi: Arama sorgusu",
        )

    def calistir(self, girdi: str) -> str:
        sorgu = girdi.lower().strip()
        for anahtar, icerik in self.BILGI_TABANI.items():
            if anahtar in sorgu or sorgu in anahtar:
                return f"Arama Sonucu [{anahtar}]: {icerik}"
        return f"Arama Sonucu: '{girdi}' sorgusu için doğrudan eşleşen bir kayıt bulunamadı."


class PythonCalistirici(TemelArac):
    """Güvenli Python kodu çalıştırma ve sonuç üretme aracı."""

    def __init__(self):
        super().__init__(
            ad="PythonCalistirici",
            aciklama="Kısa Python kodlarını çalıştırır ve çıktısını döndürür. Girdi: Python kod parçası",
        )

    def calistir(self, girdi: str) -> str:
        # Tehlikeli kütüphaneleri engelle
        yasakli = ["import os", "import sys", "subprocess", "open(", "eval(", "exec("]
        for y in yasakli:
            if y in girdi:
                return f"Güvenlik Uyarısı: '{y}' kullanımı güvenlik politikası nedeniyle engellendi."

        guvenli_builtins = {
            "sum": sum, "len": len, "min": min, "max": max, "range": range,
            "abs": abs, "round": round, "int": int, "float": float, "str": str,
            "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
        }
        yerel_alan: Dict[str, Any] = {}
        try:
            # Basit print ve matematik çalıştırma
            if "print(" in girdi:
                girdi_temiz = re.sub(r"print\((.*?)\)", r"sonuc = \1", girdi)
                exec(girdi_temiz, {"__builtins__": guvenli_builtins}, yerel_alan)
                return f"Kod Çıktısı: {yerel_alan.get('sonuc', 'Başarıyla çalıştı')}"
            else:
                exec(f"sonuc = {girdi}", {"__builtins__": guvenli_builtins}, yerel_alan)
                return f"Kod Çıktısı: {yerel_alan.get('sonuc', 'Başarıyla çalıştı')}"
        except Exception as e:
            return f"Çalıştırma Hatası: {str(e)}"


class AracKayitDefteri:
    """Tüm araçları yöneten ve ajana sunan merkezi kayıt defteri."""

    def __init__(self):
        self.araclar: Dict[str, TemelArac] = {}

    def arac_ekle(self, arac: TemelArac):
        self.araclar[arac.ad] = arac

    def arac_calistir(self, arac_adi: str, girdi: str) -> str:
        if arac_adi not in self.araclar:
            return f"Hata: '{arac_adi}' adında bir araç bulunamadı. Mevcut araçlar: {list(self.araclar.keys())}"
        return self.araclar[arac_adi].calistir(girdi)

    def sistem_promptu_aciklamasi(self) -> str:
        """Sistem promptuna eklenecek araç tanımlarını üretir."""
        satirlar = ["Kullanabileceğiniz Araçlar:"]
        for arac in self.araclar.values():
            satirlar.append(f"- {arac.ad}: {arac.aciklama}")
        return "\n".join(satirlar)
