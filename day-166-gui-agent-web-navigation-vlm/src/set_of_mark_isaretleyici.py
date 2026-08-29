"""
Set-of-Mark (SoM) Görsel İşaretleme Modülü (Day 166 - FAZ 9).
Ekran görüntüsündeki tıklanabilir buton ve elemanları numaralandırılmış etiketlerle işaretler.
"""

from typing import List, Dict, Any, Tuple


class SetOfMarkIsaretleyici:
    """Ekran Görüntüsü Elemanlarını Numaralandıran SoM İşaretleyicisi."""

    @classmethod
    def ornek_sayfa_elemanlarini_getir(cls) -> List[Dict[str, Any]]:
        """Web sayfasındaki tıklanabilir interaktif elemanları döner."""
        return [
            {"id": 1, "eleman_tipi": "Arama Giriş Kutusu (Input)", "etiket": "Google Search Input", "kutu": [320, 250, 370, 750], "merkez": (345, 500)},
            {"id": 2, "eleman_tipi": "Arama Butonu (Button)", "etiket": "Kendimi Şanslı Hissediyorum", "kutu": [400, 350, 440, 520], "merkez": (420, 435)},
            {"id": 3, "eleman_tipi": "Navigasyon Linki (Link)", "etiket": "Görseller", "kutu": [30, 850, 60, 920], "merkez": (45, 885)},
            {"id": 4, "eleman_tipi": "Oturum Aç Butonu (Button)", "etiket": "Oturum Aç", "kutu": [25, 930, 65, 990], "merkez": (45, 960)},
        ]

    @classmethod
    def eleman_etiketle(cls, eleman_listesi: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Her elemanı benzersiz ID ile indeksler."""
        return {e["id"]: e for e in eleman_listesi}
