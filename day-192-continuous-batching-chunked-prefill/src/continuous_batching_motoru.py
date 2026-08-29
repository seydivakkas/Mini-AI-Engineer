"""
Continuous Batching ve Chunked Prefill Zamanlayıcı Motoru (Day 192 - FAZ 10).
Hücresel İterasyon Seviyesinde Yığınlama (Cellular Batching) ve Prefill Dilimleme.
"""

from typing import List, Dict, Any, Optional
from enum import Enum


class IstekDurumu(Enum):
    BEKLEMEDE = "BEKLEMEDE"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    TAMAMLANDI = "TAMAMLANDI"


class LLMIstek:
    """Tekil Bir LLM Çıkarım İsteği."""

    def __init__(
        self,
        istek_id: str,
        varis_zamani: float,
        prompt_token_sayisi: int,
        hedef_uretim_token: int,
    ):
        self.istek_id = istek_id
        self.varis_zamani = varis_zamani
        self.prompt_token_sayisi = prompt_token_sayisi
        self.hedef_uretim_token = hedef_uretim_token

        self.islenen_prompt_token: int = 0
        self.uretilen_token_sayisi: int = 0
        self.durum: IstekDurumu = IstekDurumu.BEKLEMEDE

        self.ilk_token_zamani: Optional[float] = None
        self.bitis_zamani: Optional[float] = None

    @property
    def ttft(self) -> Optional[float]:
        """Time-To-First-Token (İlk Tokena Kadar Geçen Süre)."""
        if self.ilk_token_zamani is not None:
            return self.ilk_token_zamani - self.varis_zamani
        return None

    @property
    def tpot(self) -> Optional[float]:
        """Time-Per-Output-Token (Token Başına Ortalama Süre)."""
        if self.bitis_zamani is not None and self.ilk_token_zamani is not None and self.uretilen_token_sayisi > 1:
            return (self.bitis_zamani - self.ilk_token_zamani) / (self.uretilen_token_sayisi - 1)
        return None


class ContinuousBatchingScheduler:
    """
    vLLM / Orca Standardı İterasyon Seviyesinde Zamanlayıcı.
    Chunked Prefill ile Prefill ve Decode aşamalarını tek bir ileri geçişte harmanlar.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_batched_tokens: int = 512,
        chunk_size: int = 256,
    ):
        self.max_batch_size = max_batch_size
        self.max_batched_tokens = max_batched_tokens
        self.chunk_size = chunk_size

        self.kuyruktaki_istekler: List[LLMIstek] = []
        self.calisan_istekler: List[LLMIstek] = []
        self.tamamlanan_istekler: List[LLMIstek] = []

    def istek_ekle(self, istek: LLMIstek):
        """Kuyruğa yeni istek ekler."""
        self.kuyruktaki_istekler.append(istek)

    def adim_yurut(self, iterasyon_zamani: float) -> Dict[str, Any]:
        """
        Tek bir iterasyon (Forward Pass) yürütür.
        1. Aktif Decode İstekleri (Her biri 1 token)
        2. Halihazırda çalışan Prefill İsteklerinin devam dilimleri
        3. Kalan bütçeye göre Kuyruktan yeni isteklerin kabulü (Chunked Prefill)
        4. Biten isteklerin anında tahliyesi (Eviction)
        """
        # 1. Aktif Decode İsteklerini Belirle
        aktif_decode: List[LLMIstek] = [
            req for req in self.calisan_istekler if req.durum == IstekDurumu.DECODE
        ]
        harcanan_token_butcesi = len(aktif_decode)

        # 2. Çalışan Havuzdaki Mevcut Prefill İsteklerini İlerlet
        mevcut_prefill: List[LLMIstek] = [
            req for req in self.calisan_istekler if req.durum == IstekDurumu.PREFILL
        ]
        for req in mevcut_prefill:
            kalan_butce = self.max_batched_tokens - harcanan_token_butcesi
            if kalan_butce <= 0:
                break
            kalan_prompt = req.prompt_token_sayisi - req.islenen_prompt_token
            islenen_chunk = min(kalan_prompt, self.chunk_size, kalan_butce)
            req.islenen_prompt_token += islenen_chunk
            harcanan_token_butcesi += islenen_chunk

            if req.islenen_prompt_token >= req.prompt_token_sayisi:
                req.durum = IstekDurumu.DECODE
                if req.ilk_token_zamani is None:
                    req.ilk_token_zamani = iterasyon_zamani

        # 3. Kuyruktan Yeni İstek Kabul Et
        while self.kuyruktaki_istekler and len(self.calisan_istekler) < self.max_batch_size:
            kalan_butce = self.max_batched_tokens - harcanan_token_butcesi
            if kalan_butce <= 0:
                break

            aday_istek = self.kuyruktaki_istekler.pop(0)
            aday_istek.durum = IstekDurumu.PREFILL
            self.calisan_istekler.append(aday_istek)

            kalan_prompt = aday_istek.prompt_token_sayisi - aday_istek.islenen_prompt_token
            islenen_chunk = min(kalan_prompt, self.chunk_size, kalan_butce)
            aday_istek.islenen_prompt_token += islenen_chunk
            harcanan_token_butcesi += islenen_chunk

            if aday_istek.islenen_prompt_token >= aday_istek.prompt_token_sayisi:
                aday_istek.durum = IstekDurumu.DECODE
                if aday_istek.ilk_token_zamani is None:
                    aday_istek.ilk_token_zamani = iterasyon_zamani

        # 4. Decode Token Üretimi
        yeni_tamamlananlar: List[LLMIstek] = []
        for req in aktif_decode:
            req.uretilen_token_sayisi += 1
            if req.ilk_token_zamani is None:
                req.ilk_token_zamani = iterasyon_zamani

            if req.uretilen_token_sayisi >= req.hedef_uretim_token:
                req.durum = IstekDurumu.TAMAMLANDI
                req.bitis_zamani = iterasyon_zamani
                yeni_tamamlananlar.append(req)

        # 5. Bitenleri çalışan havuzundan çıkar
        for req in yeni_tamamlananlar:
            self.calisan_istekler.remove(req)
            self.tamamlanan_istekler.append(req)

        return {
            "iterasyon_zamani": iterasyon_zamani,
            "harcanan_token": harcanan_token_butcesi,
            "calisan_istek_sayisi": len(self.calisan_istekler),
            "kuyruk_uzunlugu": len(self.kuyruktaki_istekler),
            "decode_istek_sayisi": len(aktif_decode),
            "yeni_tamamlanan_sayisi": len(yeni_tamamlananlar),
        }
