"""
Llama Guard ve MLCommons Güvenlik Taksonomisi Modülü (Day 117).
LLM güvenlik ihlallerini standart S1-S6 kategorilerine ve ciddiyet derecelerine göre sınıflandırır.
"""

from typing import Dict, Any


class GuvenlikTaksonomisi:
    """Llama Guard MLCommons standart güvenlik kategorileri kataloğu."""

    KATEGORILER = {
        "S1": {
            "kod": "S1",
            "isim": "Şiddet ve Ağır Suçlar",
            "tanim": "Fiziksel şiddet, cinayet, terörizm veya patlayıcı yapımıyla ilgili talimatlar.",
            "ciddiyet": "Kritik",
            "anahtar_kelimeler": ["bomba", "patlayıcı", "silah", "suikast", "terör", "zehir"],
        },
        "S2": {
            "kod": "S2",
            "isim": "Siber Suçlar ve Zararlı Yazılım",
            "tanim": "Ransomware, keylogger, sızma yazılımı, phishing veya yetkisiz sistem erişimi.",
            "ciddiyet": "Yüksek",
            "anahtar_kelimeler": ["ransomware", "keylogger", "exploit", "trojan", "ddos", "hackle", "phishing"],
        },
        "S3": {
            "kod": "S3",
            "isim": "Cinsel Suçlar ve Taciz",
            "tanim": "Rıza dışı cinsel içerik, taciz ve şantaj materyalleri.",
            "ciddiyet": "Yüksek",
            "anahtar_kelimeler": ["şantaj", "taciz", "rıza dışı"],
        },
        "S4": {
            "kod": "S4",
            "isim": "Çocuk İstismarı ve Sömürüsü",
            "tanim": "Çocuklara yönelik her türlü istismar, zarar veya sömürü.",
            "ciddiyet": "Kritik",
            "anahtar_kelimeler": ["çocuk istismarı", "reşit olmayan"],
        },
        "S5": {
            "kod": "S5",
            "isim": "Lisanssız Yüksek Riskli Tavsiye",
            "tanim": "Yetkisiz tıbbi reçete, yasa dışı finansal manipülasyon veya hukuki temsil.",
            "ciddiyet": "Orta",
            "anahtar_kelimeler": ["kesin reçete", "ilacı bırak", "hisse manipülasyonu"],
        },
        "S6": {
            "kod": "S6",
            "isim": "PII ve Gizli Bilgi Sızıntısı",
            "tanim": "API anahtarları, şifreler, kredi kartı numaraları, TC kimlik veya gizli sistem istemi.",
            "ciddiyet": "Yüksek",
            "anahtar_kelimeler": ["api_key", "password", "sk-", "kredi kartı", "system prompt", "gizli talimat"],
        },
    }

    @classmethod
    def kategori_bilgisi(cls, kod: str) -> Dict[str, Any]:
        return cls.KATEGORILER.get(kod, {"kod": "S0", "isim": "Bilinmeyen Risk", "ciddiyet": "Düşük"})
