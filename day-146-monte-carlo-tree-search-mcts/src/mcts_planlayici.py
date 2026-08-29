"""
Monte Carlo Tree Search (MCTS) Akıl Yürütme ve Düşünce Planlayıcı Modülü (Day 146 - Faz 8).
4 Aşamalı MCTS Döngüsü: Selection, Expansion, Simulation ve Backpropagation.
"""

from typing import List, Dict, Any, Optional
from .mcts_dugumu import MCTSDugumu
from .rollout_politika_motoru import RolloutPolitikaMotoru


class MCTSDusuncePlanlayici:
    """LLM muhakeme adımları ve kombinatoryal problemler için MCTS planlama motoru."""

    def __init__(self, hedef: float = 24.0, c_kesif: float = 0.8, simulasyon_sayisi: int = 250):
        self.hedef = hedef
        self.c_kesif = c_kesif
        self.simulasyon_sayisi = simulasyon_sayisi
        self.dugum_sayaci = 0

    def _cocuk_dugumleri_turet(self, dugum: MCTSDugumu) -> List[MCTSDugumu]:
        """Düğümün kalan sayılarından geçerli 4 işlemle çocuk düğümler türetir."""
        sayilar = dugum.sayilar
        if len(sayilar) < 2:
            return []

        cocuklar = []
        n = len(sayilar)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = sayilar[i], sayilar[j]
                digerleri = [sayilar[k] for k in range(n) if k != i and k != j]

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
                    yeni_sayilar = digerleri + [sonuc]
                    cocuk = MCTSDugumu(
                        durum_id=f"mcts_{self.dugum_sayaci}",
                        sayilar=yeni_sayilar,
                        adim_gecmisi=dugum.adim_gecmisi + [adim_metni],
                        ebeveyn=dugum,
                    )
                    cocuklar.append(cocuk)

        return cocuklar

    def planla(self, baslangic_sayilari: List[float]) -> Dict[str, Any]:
        """
        Başlangıç sayılarından 4 aşamalı MCTS döngüsünü çalıştırır ve en iyi düşünce yolunu döner.
        """
        self.dugum_sayaci = 1
        kok = MCTSDugumu(
            durum_id="mcts_root",
            sayilar=baslangic_sayilari,
            adim_gecmisi=[],
        )
        kok.cocuklar = self._cocuk_dugumleri_turet(kok)

        for _ in range(self.simulasyon_sayisi):
            # 1. SELECTION (Seçim): Kökten başlayarak UCT ile in
            mevcut = kok
            while mevcut.cocuklar and not mevcut.terminal_mi():
                ziyaretsiz = [c for c in mevcut.cocuklar if c.ziyaret_sayisi == 0]
                if ziyaretsiz:
                    mevcut = ziyaretsiz[0]
                    break
                else:
                    mevcut = mevcut.en_iyi_cocugu_sec(self.c_kesif)

            # 2. EXPANSION (Genişletme): Eğer yaprak düğümse ve terminal değilse genişlet
            if not mevcut.cocuklar and not mevcut.terminal_mi():
                mevcut.cocuklar = self._cocuk_dugumleri_turet(mevcut)
                if mevcut.cocuklar:
                    hedef_c = next((c for c in mevcut.cocuklar if len(c.sayilar) == 1 and abs(c.sayilar[0] - self.hedef) < 1e-4), None)
                    mevcut = hedef_c if hedef_c else mevcut.cocuklar[0]

            # 3. SIMULATION (Rollout)
            odul = RolloutPolitikaMotoru.simule_et(mevcut.sayilar, self.hedef)

            # 4. BACKPROPAGATION
            gezen = mevcut
            while gezen is not None:
                gezen.ziyaret_sayisi += 1
                gezen.toplam_odul += odul
                gezen = gezen.ebeveyn

        # Nihai Karar: Ağaç içinden hedefe ulaşan veya en yüksek Q'lu tam yolu bul
        en_iyi_yol = []
        gezen = kok
        while gezen.cocuklar:
            hedef_c = next((c for c in gezen.cocuklar if len(c.sayilar) == 1 and abs(c.sayilar[0] - self.hedef) < 1e-4), None)
            if hedef_c:
                gezen = hedef_c
                en_iyi_yol.append(gezen)
                break
            adaylar = [c for c in gezen.cocuklar if c.ziyaret_sayisi > 0]
            if not adaylar:
                break
            gezen = max(adaylar, key=lambda c: (c.ortalama_deger, c.ziyaret_sayisi))
            en_iyi_yol.append(gezen)
            if len(gezen.sayilar) == 1 and abs(gezen.sayilar[0] - self.hedef) < 1e-4:
                break

        son_dugum = en_iyi_yol[-1] if en_iyi_yol else kok
        cozum_bulundu = False
        if len(son_dugum.sayilar) == 1 and abs(son_dugum.sayilar[0] - self.hedef) < 1e-4:
            cozum_bulundu = True

        return {
            "algoritma": "Monte Carlo Tree Search (MCTS + UCT)",
            "cozum_bulundu_mu": cozum_bulundu,
            "en_iyi_yol_adimlari": son_dugum.adim_gecmisi,
            "nihai_sayi": son_dugum.sayilar[0] if son_dugum.sayilar else None,
            "toplam_simulasyon": self.simulasyon_sayisi,
            "toplam_kesfedilen_dugum": self.dugum_sayaci,
            "kok_ziyaret_sayisi": kok.ziyaret_sayisi,
            "kok_ortalama_q": round(kok.ortalama_deger, 4),
        }
