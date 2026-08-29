"""
Kod Üreticisi ve Otomatik Onarım Modülü (Day 151 - Faz 8).
LLM tabanlı TDD kod üretimi ve hata yığınına (Traceback) dayalı kod onarımı.
"""

from typing import Dict, Any


class KodUreticisi:
    """TDD yaklaşımıyla kod üreten ve hata raporlarına göre yamalayan ajan."""

    def __init__(self, model_adi: str = "MiniCodeGen-8B"):
        self.model_adi = model_adi

    def ilk_kodu_uret(self, gorev: str) -> Dict[str, Any]:
        """
        İlk taslak kodu üretir. (Sınır durumu hatası içerir: boş string veya son karakter tamponu).
        """
        # Problem: Run-Length Encoding (RLE) -> "AABBC" => "A2B2C1"
        hatali_kod = (
            "def run_length_encoding(metin: str) -> str:\n"
            "    # Hatalı: Boş string kontrolü yok, IndexError riski!\n"
            "    sonuc = []\n"
            "    mevcut = metin[0]\n"
            "    sayac = 1\n"
            "    for c in metin[1:]:\n"
            "        if c == mevcut:\n"
            "            sayac += 1\n"
            "        else:\n"
            "            sonuc.append(f'{mevcut}{sayac}')\n"
            "            mevcut = c\n"
            "            sayac = 1\n"
            "    # Hatalı: Son karakter grubunu eklemeyi unuttu!\n"
            "    return ''.join(sonuc)\n"
        )

        return {
            "kod": hatali_kod,
            "aciklama": "İlk taslak RLE fonksiyonu oluşturuldu.",
            "tur": 1,
        }

    def kodu_onar(self, gorev: str, onceki_kod: str, hata_raporu: str, tur: int) -> Dict[str, Any]:
        """
        PyTest terminal hata raporunu (Traceback) analiz ederek kodu yamalar.
        """
        onarma_monologu = (
            f"<think>\n"
            f"PyTest Hata Analizi: {hata_raporu[:150]}...\n"
            f"1. metin='' boş olduğunda metin[0] IndexError veriyor. Boş string koruması eklenmeli.\n"
            f"2. Döngü bittiğinde son karakter grubu ('mevcut' + 'sayac') sonuca eklenmiyor.\n"
            f"Düzeltme uygulanıyor.\n"
            f"</think>"
        )

        duzeltilmis_kod = (
            "def run_length_encoding(metin: str) -> str:\n"
            "    if not metin:\n"
            "        return ''\n"
            "    sonuc = []\n"
            "    mevcut = metin[0]\n"
            "    sayac = 1\n"
            "    for c in metin[1:]:\n"
            "        if c == mevcut:\n"
            "            sayac += 1\n"
            "        else:\n"
            "            sonuc.append(f'{mevcut}{sayac}')\n"
            "            mevcut = c\n"
            "            sayac = 1\n"
            "    sonuc.append(f'{mevcut}{sayac}')  # Son grup eklendi\n"
            "    return ''.join(sonuc)\n"
        )

        return {
            "kod": duzeltilmis_kod,
            "onarma_monologu": onarma_monologu,
            "aciklama": "Boş string kontrolü ve son grup tamponu eklendi.",
            "tur": tur,
        }
