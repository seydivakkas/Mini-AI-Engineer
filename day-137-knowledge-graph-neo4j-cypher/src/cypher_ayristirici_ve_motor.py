"""
Cypher Ayrıştırıcı ve Yürütme Motoru (Cypher Engine) Modülü (Day 137 - Faz 7 - GraphRAG-2).
Deklaratif Cypher şablonlarını (MATCH ... WHERE ... RETURN) ayrıştırıp graf deposunda icra eden motor.
"""

from typing import List, Dict, Any, Optional
import re

from .ozellikli_graf_deposu import OzellikliGrafDeposu, Dugum, Kenar


class CypherMotoru:
    """Cypher deklaratif sorgu dilini ayrıştırıp LPG üzerinde icra eden motor."""

    def __init__(self, depo: OzellikliGrafDeposu):
        self.depo = depo

    def sorgula(self, cypher_sorgusu: str) -> List[Dict[str, Any]]:
        """
        Desteklenen Cypher Kalıpları:
        1. MATCH (a)-[r:ILISKI_TIPI]->(b) WHERE a.id = '...' RETURN b
        2. MATCH (n:ETIKET) WHERE ... RETURN n
        3. MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a.id = '...' RETURN c
        """
        sorgu = cypher_sorgusu.strip()

        # 1. Kalıp: 2-Hop Zinciri -> MATCH (a)-[r1]->(b)-[r2]->(c) WHERE a.id = 'X' RETURN c
        iki_hop_match = re.search(
            r"MATCH\s*\(([a-zA-Z0-9_]+)\)-\[([^\]]*)\]->\(([a-zA-Z0-9_]+)\)-\[([^\]]*)\]->\(([a-zA-Z0-9_]+)\)\s*WHERE\s*([a-zA-Z0-9_]+)\.(id|etiket|isim)\s*=\s*'([^']+)'\s*RETURN\s*([a-zA-Z0-9_,\s]+)",
            sorgu,
            re.IGNORECASE,
        )
        if iki_hop_match:
            filtre_alan = iki_hop_match.group(7).lower()
            baslangic_deger = iki_hop_match.group(8)
            sonuclar = []

            # Eşleşen başlangıç düğümlerini bul
            baslangic_dugumleri = []
            for d in self.depo.tum_dugumler():
                if filtre_alan == "id" and d.id == baslangic_deger:
                    baslangic_dugumleri.append(d)
                elif filtre_alan == "etiket" and d.etiket == baslangic_deger:
                    baslangic_dugumleri.append(d)

            for b_dugum in baslangic_dugumleri:
                # 1. Hop
                for k1, dugum_b in self.depo.komsulari_getir(b_dugum.id, yon="OUT"):
                    # 2. Hop
                    for k2, dugum_c in self.depo.komsulari_getir(dugum_b.id, yon="OUT"):
                        sonuclar.append({
                            "baslangic": b_dugum.id,
                            "ara_dugum": dugum_b.id,
                            "hedef_dugum": dugum_c.id,
                            "yol": f"({b_dugum.id}) -[{k1.iliski_tipi}]-> ({dugum_b.id}) -[{k2.iliski_tipi}]-> ({dugum_c.id})",
                        })
            return sonuclar

        # 2. Kalıp: 1-Hop İlişki -> MATCH (a)-[r:TIP]->(b) WHERE a.id = 'X' RETURN b
        bir_hop_match = re.search(
            r"MATCH\s*\(([a-zA-Z0-9_]+)\)(?:-\[([a-zA-Z0-9_:]*)\]->\(([a-zA-Z0-9_]+)\))?\s*(?:WHERE\s*([a-zA-Z0-9_]+)\.(id|etiket|isim)\s*=\s*'([^']+)')?\s*RETURN\s*([a-zA-Z0-9_,\s]+)",
            sorgu,
            re.IGNORECASE,
        )
        if bir_hop_match:
            kaynak_degisken = bir_hop_match.group(1)
            iliski_tanimi = bir_hop_match.group(2) or ""
            hedef_degisken = bir_hop_match.group(3)
            filtre_alan = bir_hop_match.group(5)
            filtre_deger = bir_hop_match.group(6)

            istenen_iliski = ""
            if ":" in iliski_tanimi:
                istenen_iliski = iliski_tanimi.split(":")[-1].strip()

            sonuclar = []
            # Tüm düğümler üzerinde gezin
            for dugum in self.depo.tum_dugumler():
                # Filtre kontrolü
                if filtre_alan and filtre_deger:
                    if filtre_alan == "id" and dugum.id != filtre_deger:
                        continue
                    elif filtre_alan == "etiket" and dugum.etiket != filtre_deger:
                        continue

                # Eğer sadece düğüm sorgulanıyorsa
                if not hedef_degisken:
                    sonuclar.append({"dugum": dugum.id, "etiket": dugum.etiket, "ozellikler": dugum.ozellikler})
                    continue

                # İlişki kenarları üzerinde gezin
                for kenar, komsu in self.depo.komsulari_getir(dugum.id, yon="OUT"):
                    if istenen_iliski and kenar.iliski_tipi != istenen_iliski:
                        continue
                    sonuclar.append({
                        "kaynak": dugum.id,
                        "iliski": kenar.iliski_tipi,
                        "hedef": komsu.id,
                        "agirlik": kenar.agirlik,
                        "hedef_etiket": komsu.etiket,
                    })

            return sonuclar

        # Varsayılan Fallback: Tüm Düğümleri Döndür
        return [{"id": d.id, "etiket": d.etiket} for d in self.depo.tum_dugumler()]
