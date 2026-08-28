"""
40.1: Başlık ve Paragraf Duyarlı Semantik Metin Parçalayıcı (Header-Aware Semantic Chunker).
"""

from typing import List, Dict, Any
import re


class SemantikMetinParcalayici:
    """Teknik dokümanları alt başlık, paragraf bütünlüğü ve örtüşme (overlap) koruyarak parçalar."""

    def __init__(self, max_karakter: int = 400, overlap_karakter: int = 60):
        self.max_karakter = max_karakter
        self.overlap = overlap_karakter

    def dokuman_parcala(self, dokuman: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Tek bir teknik dokümanı semantik paragraflara böler ve metadata zenginleştirmesi yapar."""
        ham_metin = dokuman.get("metin", "").strip()
        paragraf_bloklari = re.split(r"\n\s*\n", ham_metin)

        parcalar = []
        for blok in paragraf_bloklari:
            blok = blok.strip()
            if not blok:
                continue

            # Başlık tespiti (Örn: "1.1 İplik Numaralandırma:")
            baslik_eslesme = re.match(r"^(\d+\.\d+\s+[^:\n]+):", blok)
            if baslik_eslesme:
                alt_baslik = baslik_eslesme.group(1).strip()
            else:
                alt_baslik = dokuman.get("baslik", "Genel Bölüm")

            # Eğer blok belirlenen boyuttan küçükse doğrudan chunk yap
            if len(blok) <= self.max_karakter:
                parcalar.append({
                    "chunk_id": f"CHUNK-{dokuman['dokuman_id']}-{len(parcalar)+1:02d}",
                    "dokuman_id": dokuman["dokuman_id"],
                    "ana_baslik": dokuman["baslik"],
                    "alt_baslik": alt_baslik,
                    "kategori": dokuman.get("kategori", "genel"),
                    "kaynak_standart": dokuman.get("kaynak_standart", ""),
                    "guven_derecesi": dokuman.get("guven_derecesi", "BILINMIYOR"),
                    "metin": blok,
                    "karakter_uzunlugu": len(blok),
                    "kelime_sayisi": len(blok.split())
                })
            else:
                # Sliding window overlap
                adim = self.max_karakter - self.overlap
                for i in range(0, len(blok), adim):
                    dilim = blok[i:i + self.max_karakter]
                    if len(dilim.strip()) < 40:
                        continue
                    parcalar.append({
                        "chunk_id": f"CHUNK-{dokuman['dokuman_id']}-{len(parcalar)+1:02d}",
                        "dokuman_id": dokuman["dokuman_id"],
                        "ana_baslik": dokuman["baslik"],
                        "alt_baslik": alt_baslik,
                        "kategori": dokuman.get("kategori", "genel"),
                        "kaynak_standart": dokuman.get("kaynak_standart", ""),
                        "guven_derecesi": dokuman.get("guven_derecesi", "BILINMIYOR"),
                        "metin": dilim.strip(),
                        "karakter_uzunlugu": len(dilim.strip()),
                        "kelime_sayisi": len(dilim.strip().split())
                    })

        return parcalar

    def korpus_parcala(self, korpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tüm korpusu parçalayarak düzleştirilmiş chunk listesi döndürür."""
        tum_parcalar = []
        for dok in korpus:
            tum_parcalar.extend(self.dokuman_parcala(dok))
        return tum_parcalar
