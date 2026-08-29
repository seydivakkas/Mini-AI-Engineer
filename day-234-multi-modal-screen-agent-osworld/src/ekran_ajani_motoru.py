"""
Çok Modlu Ekran Ajanı Motoru (Computer Use / OSWorld) (Day 234 - FAZ 12).
Piksel Koordinat Eşleme (Visual Grounding), Fare/Klavye İlkel Eylemleri.
"""

from typing import Dict, Any, List, Optional, Tuple


class ScreenElement:
    """Ekrandaki Görsel Bileşen ve Koordinat Modeli."""

    def __init__(
        self,
        element_id: int,
        etiket: str,
        x: int,
        y: int,
        genislik: int = 50,
        yukseklik: int = 30,
    ):
        self.element_id = element_id
        self.etiket = etiket
        self.x = x
        self.y = y
        self.genislik = genislik
        self.yukseklik = yukseklik

    def merkez_koordinati(self) -> Tuple[int, int]:
        """Tıklanabilir merkez koordinatını (x_c, y_c) döndürür."""
        return (self.x + self.genislik // 2, self.y + self.yukseklik // 2)


class GUIAction:
    """İşletim Sistemi Seviyesi İlkel Fare ve Klavye Eylemi."""

    def __init__(
        self,
        eylem_turu: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        metin: Optional[str] = None,
        tuslar: Optional[List[str]] = None,
    ):
        self.eylem_turu = eylem_turu.upper()
        self.x = x
        self.y = y
        self.metin = metin
        self.tuslar = tuslar

    def format_metni(self) -> str:
        if self.eylem_turu in ["CLICK", "DOUBLE_CLICK"]:
            return f"🖱️ {self.eylem_turu}(x={self.x}, y={self.y})"
        elif self.eylem_turu == "TYPE":
            return f"⌨️ TYPE(text='{self.metin}')"
        elif self.eylem_turu == "HOTKEY":
            return f"⌨️ HOTKEY({'+'.join(self.tuslar or [])})"
        elif self.eylem_turu == "SCROLL":
            return f"📜 SCROLL({self.metin})"
        return f"🖥️ ACTION({self.eylem_turu})"


class ComputerUseAgent:
    """Masaüstü Ekranını Okuyup Fare ve Klavye Yöneten Çok Modlu Ajan."""

    def __init__(self, cozunurluk: Tuple[int, int] = (1920, 1080)):
        self.cozunurluk = cozunurluk
        self.aktif_bilesenler: Dict[str, ScreenElement] = {}
        self.fare_konumu: Tuple[int, int] = (0, 0)
        self.islem_gecmisi: List[str] = []

    def ekrana_bilesen_ekle(self, bilesen: ScreenElement):
        self.aktif_bilesenler[bilesen.etiket.lower()] = bilesen

    def bilesen_bul(self, hedef_etiket: str) -> Optional[ScreenElement]:
        """Ekrandaki etikete göre bileşeni ve merkez koordinatını bulur."""
        return self.aktif_bilesenler.get(hedef_etiket.lower())

    def eylem_icra_et(self, eylem: GUIAction) -> Dict[str, Any]:
        """İlkel GUI eylemini işletim sistemi simülasyonunda koşturur."""
        if eylem.eylem_turu in ["CLICK", "DOUBLE_CLICK"]:
            self.fare_konumu = (eylem.x or 0, eylem.y or 0)
            log = f"{eylem.format_metni()} -> Fare taşındı ve tıklandı."
        elif eylem.eylem_turu == "TYPE":
            log = f"{eylem.format_metni()} -> Metin aktif alana yazıldı."
        elif eylem.eylem_turu == "HOTKEY":
            log = f"{eylem.format_metni()} -> Kısayol kombinasyonu tetiklendi."
        else:
            log = f"{eylem.format_metni()} -> Eylem icra edildi."

        self.islem_gecmisi.append(log)
        return {
            "durum": "BASARILI",
            "eylem": eylem.eylem_turu,
            "log": log,
            "fare_konumu": self.fare_konumu,
        }

    def gorevi_tamamla(self, gorev_plani: List[GUIAction]) -> List[Dict[str, Any]]:
        """Çok adımlı görev planını sırayla icra eder."""
        sonuclar = []
        for eylem in gorev_plani:
            sonuclar.append(self.eylem_icra_et(eylem))
        return sonuclar
