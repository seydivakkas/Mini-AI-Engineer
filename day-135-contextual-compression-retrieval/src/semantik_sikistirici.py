"""
Semantik Bağlam Sıkıştırıcı Modülü (Day 135 - Faz 7).
Cümlelerin soruyla olan anlamsal uygunluğunu ölçüp alakasız gürültüleri eleyen motor.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .baglam_ayristirici import CumleBirimi, BaglamAyristirici


class SemantikBaglamSikistirici:
    """Cümle düzeyinde anlamsal puanlama ve budama ile bağlamı sıkıştıran motor."""

    def __init__(self, vektor_boyutu: int = 128, esik_skoru: float = 0.30):
        self.vektor_boyutu = vektor_boyutu
        self.esik_skoru = esik_skoru

    def _metin_vektorlestir(self, metin: str) -> torch.Tensor:
        """Deterministik ve anlamsal L2 normalize embedding üretir."""
        sozcukler = metin.lower().split()
        vektor = np.zeros(self.vektor_boyutu, dtype=np.float32)

        for idx, kelime in enumerate(sozcukler):
            np.random.seed(abs(hash(kelime)) % (2**31))
            agirlik = 1.0 / (idx + 1) ** 0.4
            vektor += np.random.randn(self.vektor_boyutu).astype(np.float32) * agirlik

        tensör = torch.tensor(vektor, dtype=torch.float32).unsqueeze(0)
        return F.normalize(tensör, p=2, dim=1)

    def cumleleri_puanla(
        self, sorgu: str, cumleler: List[CumleBirimi]
    ) -> List[Tuple[CumleBirimi, float]]:
        """Sorgu ile her cümlenin kosinüs benzerliğini hesaplar."""
        if not cumleler:
            return []

        sorgu_emb = self._metin_vektorlestir(sorgu)
        cumle_embler = [self._metin_vektorlestir(c.metin) for c in cumleler]
        cumle_tensör = torch.cat(cumle_embler, dim=0)

        skorlar = F.cosine_similarity(sorgu_emb, cumle_tensör).tolist()
        return [(c, round(float(s), 4)) for c, s in zip(cumleler, skorlar)]

    def sikistir(
        self, sorgu: str, belgeler: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Belgeleri ayrıştırır, alakasız cümleleri eler ve sıkıştırılmış bağlam üretir.
        """
        cumle_birimleri = BaglamAyristirici.toplu_ayristir(belgeler)
        puanli_cumleler = self.cumleleri_puanla(sorgu, cumle_birimleri)

        ham_token_toplami = sum(c.token_tahmini for c in cumle_birimleri)
        ham_karakter_toplami = sum(c.karakter_sayisi for c in cumle_birimleri)

        # Eşik üstü kalan cümleleri filtrele
        secilen_cumleler: List[CumleBirimi] = []
        elenen_cumleler: List[CumleBirimi] = []
        tum_skorlar = []

        for cumle, skor in puanli_cumleler:
            tum_skorlar.append(skor)
            if skor >= self.esik_skoru:
                secilen_cumleler.append(cumle)
            else:
                elenen_cumleler.append(cumle)

        # En az 1 cümle kalmasını garanti et (Fallback: En yüksek skorlu cümle)
        if not secilen_cumleler and puanli_cumleler:
            en_iyi_cumle, en_iyi_skor = max(puanli_cumleler, key=lambda x: x[1])
            secilen_cumleler.append(en_iyi_cumle)
            if en_iyi_cumle in elenen_cumleler:
                elenen_cumleler.remove(en_iyi_cumle)

        sikistirilmis_token_toplami = sum(c.token_tahmini for c in secilen_cumleler)
        sikistirilmis_karakter_toplami = sum(c.karakter_sayisi for c in secilen_cumleler)

        tasarruf_orani = (
            (1.0 - (sikistirilmis_token_toplami / max(1, ham_token_toplami))) * 100.0
        )

        # Belge bazında birleştirilmiş bağlam
        belge_gruplari: Dict[str, List[str]] = {}
        for c in secilen_cumleler:
            belge_gruplari.setdefault(c.doc_id, []).append(c.metin)

        nihai_baglam_bloklari = []
        for doc_id, metin_listesi in belge_gruplari.items():
            nihai_baglam_bloklari.append(f"[KAYNAK: {doc_id}]\n" + " ".join(metin_listesi))

        nihai_baglam = "\n\n---\n\n".join(nihai_baglam_bloklari)

        return {
            "sorgu": sorgu,
            "esik_skoru": self.esik_skoru,
            "toplam_cumle_sayisi": len(cumle_birimleri),
            "secilen_cumle_sayisi": len(secilen_cumleler),
            "elenen_cumle_sayisi": len(elenen_cumleler),
            "ham_token": ham_token_toplami,
            "sikistirilmis_token": sikistirilmis_token_toplami,
            "token_tasarrufu_yuzde": round(tasarruf_orani, 2),
            "ham_karakter": ham_karakter_toplami,
            "sikistirilmis_karakter": sikistirilmis_karakter_toplami,
            "puanli_cumleler": puanli_cumleler,
            "nihai_baglam": nihai_baglam,
        }
