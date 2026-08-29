"""
Tree of Thoughts (ToT) Arama Motoru Modülü (Day 144 - Faz 8).
Genişlik Öncelikli Arama (BFS), Derinlik Öncelikli Arama (DFS) ve Geri İzleme (Backtracking) algoritmaları.
"""

from typing import List, Dict, Any, Optional
import itertools

from .dusunce_durumu import DusunceDurumu
from .durum_degerlendirici import DurumDegerlendirici


class TreeOfThoughtsMotoru:
    """Game of 24 ve planlama problemleri için Tree of Thoughts arama motoru."""

    def __init__(self, hedef: float = 24.0, maks_genislik: int = 5):
        self.hedef = hedef
        self.maks_genislik = maks_genislik
        self.dugum_sayaci = 0

    def _cocuk_durumlari_uret(self, durum: DusunceDurumu) -> List[DusunceDurumu]:
        """Kalan sayılardan 2 tanesini seçip 4 temel işlemle yeni düşünce durumları türetir."""
        sayilar = durum.sayilar
        if len(sayilar) < 2:
            return []

        cocuklar = []
        n = len(sayilar)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = sayilar[i], sayilar[j]
                kalan_digerleri = [sayilar[k] for k in range(n) if k != i and k != j]

                islemler = [
                    (a + b, f"{a:g} + {b:g} = {a+b:g}"),
                    (a - b, f"{a:g} - {b:g} = {a-b:g}"),
                    (b - a, f"{b:g} - {a:g} = {b-a:g}"),
                    (a * b, f"{a:g} * {b:g} = {a*b:g}"),
                ]
                if abs(b) > 1e-5:
                    islemler.append((a / b, f"{a:g} / {b:g} = {a/b:g}"))
                if abs(a) > 1e-5:
                    islemler.append((b / a, f"{b:g} / {a:g} = {b/a:g}"))

                for sonuc, adim_metni in islemler:
                    self.dugum_sayaci += 1
                    yeni_sayilar = kalan_digerleri + [sonuc]
                    yeni_durum = DusunceDurumu(
                        durum_id=f"node_{self.dugum_sayaci}",
                        sayilar=yeni_sayilar,
                        adim_gecmisi=durum.adim_gecmisi + [adim_metni],
                        ebeveyn_id=durum.durum_id,
                        derinlik=durum.derinlik + 1,
                    )
                    puan, etiket = DurumDegerlendirici.degerlendir(yeni_durum, self.hedef)
                    yeni_durum.deger_puani = puan
                    yeni_durum.degerlendirme = etiket
                    cocuklar.append(yeni_durum)

        return cocuklar

    def bfs_ara(self, baslangic_sayilari: List[float], beam_genisligi: int = 3) -> Dict[str, Any]:
        """
        Genişlik Öncelikli Arama (BFS): Katman katman en iyi k durumu genişletir.
        """
        self.dugum_sayaci = 1
        kok = DusunceDurumu(
            durum_id="node_1",
            sayilar=baslangic_sayilari,
            adim_gecmisi=[],
            derinlik=0,
        )
        kok.deger_puani, kok.degerlendirme = DurumDegerlendirici.degerlendir(kok, self.hedef)

        katman: List[DusunceDurumu] = [kok]
        toplam_kesfedilen = 1
        toplam_budanan = 0
        cozum_durumu: Optional[DusunceDurumu] = None

        while katman and not cozum_durumu:
            yeni_adaylar: List[DusunceDurumu] = []
            for d in katman:
                if d.hedefe_ulasti_mi(self.hedef):
                    cozum_durumu = d
                    break
                cocuklar = self._cocuk_durumlari_uret(d)
                toplam_kesfedilen += len(cocuklar)

                # İmkansız durumları buda (Pruning)
                gecerliler = []
                for c in cocuklar:
                    if c.degerlendirme == "imkansiz":
                        toplam_budanan += 1
                    else:
                        gecerliler.append(c)
                yeni_adaylar.extend(gecerliler)

            if cozum_durumu or not yeni_adaylar:
                break

            # En iyi beam_genisligi kadar durumu seç (Beam Search / Top-K)
            yeni_adaylar.sort(key=lambda x: x.deger_puani, reverse=True)
            katman = yeni_adaylar[:beam_genisligi]

        return {
            "algoritma": "Tree of Thoughts (BFS)",
            "cozum_bulundu_mu": cozum_durumu is not None,
            "adim_gecmisi": cozum_durumu.adim_gecmisi if cozum_durumu else [],
            "toplam_kesfedilen_dugum": toplam_kesfedilen,
            "toplam_budanan_dugum": toplam_budanan,
            "nihai_sayi": cozum_durumu.sayilar[0] if cozum_durumu else None,
        }

    def dfs_ara(self, baslangic_sayilari: List[float], maks_derinlik: int = 4) -> Dict[str, Any]:
        """
        Derinlik Öncelikli Arama (DFS) ve Geri İzleme (Backtracking).
        """
        self.dugum_sayaci = 1
        kok = DusunceDurumu(
            durum_id="node_1",
            sayilar=baslangic_sayilari,
            adim_gecmisi=[],
            derinlik=0,
        )

        toplam_kesfedilen = 1
        toplam_budanan = 0
        geri_izleme_sayisi = 0
        cozum_durumu: Optional[DusunceDurumu] = None

        yigit: List[DusunceDurumu] = [kok]

        while yigit and not cozum_durumu:
            mevcut = yigit.pop()

            if mevcut.hedefe_ulasti_mi(self.hedef):
                cozum_durumu = mevcut
                break

            if mevcut.derinlik >= maks_derinlik:
                geri_izleme_sayisi += 1
                continue

            cocuklar = self._cocuk_durumlari_uret(mevcut)
            toplam_kesfedilen += len(cocuklar)

            gecerliler = []
            for c in cocuklar:
                if c.degerlendirme == "imkansiz":
                    toplam_budanan += 1
                else:
                    gecerliler.append(c)

            if not gecerliler:
                geri_izleme_sayisi += 1
            else:
                # Yüksek puanlılar yığının en üstünde kalsın diye küçükten büyüğe sıralayıp ekle
                gecerliler.sort(key=lambda x: x.deger_puani)
                yigit.extend(gecerliler)

        return {
            "algoritma": "Tree of Thoughts (DFS + Backtracking)",
            "cozum_bulundu_mu": cozum_durumu is not None,
            "adim_gecmisi": cozum_durumu.adim_gecmisi if cozum_durumu else [],
            "toplam_kesfedilen_dugum": toplam_kesfedilen,
            "toplam_budanan_dugum": toplam_budanan,
            "geri_izleme_sayisi": geri_izleme_sayisi,
            "nihai_sayi": cozum_durumu.sayilar[0] if cozum_durumu else None,
        }
