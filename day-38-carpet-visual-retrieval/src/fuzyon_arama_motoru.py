"""
Çoklu Özellik Füzyonu ve Görsel Halı Arama Motoru (Multi-Feature Visual Retrieval Engine).
Renk ve GLCM/LBP Doku Özelliklerinin Ağırlıklı Hibrit Eşleşmesi.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from .renk_cikarici import RenkOzellikCikarici
from .doku_cikarici import DokuOzellikCikarici


class CokluOzellikFuzyonAramaMotoru:
    """
    Halı ve kumaş görsellerini Renk (HSV+Moment) ve Doku (GLCM+LBP) vektörlerine ayrıştırarak
    ağırlıklı hibrit benzerlik araması (Top-K Visual Retrieval) yapar.
    """

    def __init__(
        self,
        renk_agirligi: float = 0.55,
        doku_agirligi: float = 0.45,
        renk_cikarici: Optional[RenkOzellikCikarici] = None,
        doku_cikarici: Optional[DokuOzellikCikarici] = None
    ):
        toplam_agirlik = renk_agirligi + doku_agirligi
        self.w_renk = renk_agirligi / (toplam_agirlik + 1e-12)
        self.w_doku = doku_agirligi / (toplam_agirlik + 1e-12)

        self.renk_cikarici = renk_cikarici or RenkOzellikCikarici()
        self.doku_cikarici = doku_cikarici or DokuOzellikCikarici()

        self.indeks: List[Dict[str, Any]] = []

    def _kosinus_benzerligi(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """İki normalize vektör arasındaki kosinüs benzerliği [-1, 1] -> [0, 1]."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        skor = dot / (norm1 * norm2)
        return float(np.clip((skor + 1.0) / 2.0, 0.0, 1.0))

    def katalog_indeksle(self, katalog: List[Dict[str, Any]]) -> None:
        """Katalogdaki tüm halıların renk ve doku özelliklerini çıkarıp belleğe indeksler."""
        self.indeks = []
        for item in katalog:
            gorsel = item["gorsel"]
            r_oz = self.renk_cikarici.cikar(gorsel)
            d_oz = self.doku_cikarici.cikar(gorsel)

            self.indeks.append({
                "id": item["id"],
                "baslik": item["baslik"],
                "kategori": item["kategori"],
                "iplik_tipi": item.get("iplik_tipi", ""),
                "hav_yuksekligi": item.get("hav_yuksekligi", ""),
                "ana_renkler": item.get("ana_renkler", []),
                "gorsel": gorsel,
                "renk_vektoru": r_oz["renk_vektoru"],
                "doku_vektoru": d_oz["doku_vektoru"],
                "haralick": d_oz["haralick_ortalama"],
                "ortalama_rgb": r_oz["ortalama_rgb"]
            })

    def gorsel_ara(
        self,
        sorgu_gorseli: Image.Image,
        top_k: int = 3,
        ozel_renk_agirligi: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Sorgu halı görseli ile katalog arasında çoklu özellik hibrit araması yapar.
        """
        if not self.indeks:
            raise ValueError("Katalog indeksi boş! Lütfen önce `katalog_indeksle()` çağırın.")

        w_r = self.w_renk if ozel_renk_agirligi is None else ozel_renk_agirligi
        w_d = 1.0 - w_r

        # Sorgu özelliklerini çıkar
        sorgu_renk = self.renk_cikarici.cikar(sorgu_gorseli)
        sorgu_doku = self.doku_cikarici.cikar(sorgu_gorseli)

        v_q_renk = sorgu_renk["renk_vektoru"]
        v_q_doku = sorgu_doku["doku_vektoru"]

        sonuclar = []
        for item in self.indeks:
            s_renk = self._kosinus_benzerligi(v_q_renk, item["renk_vektoru"])
            s_doku = self._kosinus_benzerligi(v_q_doku, item["doku_vektoru"])

            # Ağırlıklı Füzyon Skoru
            s_toplam = (w_r * s_renk) + (w_d * s_doku)

            sonuclar.append({
                "id": item["id"],
                "baslik": item["baslik"],
                "kategori": item["kategori"],
                "iplik_tipi": item["iplik_tipi"],
                "hav_yuksekligi": item["hav_yuksekligi"],
                "ana_renkler": item["ana_renkler"],
                "gorsel": item["gorsel"],
                "hibrit_skor": float(round(s_toplam * 100.0, 2)),
                "renk_skor": float(round(s_renk * 100.0, 2)),
                "doku_skor": float(round(s_doku * 100.0, 2)),
                "haralick": item["haralick"]
            })

        # Skora göre büyükten küçüğe sırala
        sonuclar.sort(key=lambda x: x["hibrit_skor"], reverse=True)
        top_sonuclar = sonuclar[:top_k]

        return {
            "sorgu_ozellikleri": {
                "haralick": sorgu_doku["haralick_ortalama"],
                "ortalama_rgb": sorgu_renk["ortalama_rgb"],
                "hsv_histogram": sorgu_renk["hsv_histogram"],
                "lbp_histogram": sorgu_doku["lbp_histogram"]
            },
            "kullanilan_agirliklar": {
                "renk_agirligi": float(round(w_r, 2)),
                "doku_agirligi": float(round(w_d, 2))
            },
            "toplam_katalog_boyutu": len(self.indeks),
            "getirilen_sayi": len(top_sonuclar),
            "sonuclar": top_sonuclar
        }
