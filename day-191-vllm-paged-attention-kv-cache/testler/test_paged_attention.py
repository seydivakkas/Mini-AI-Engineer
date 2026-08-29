"""
PyTest Birim Testleri - Day 191: vLLM PagedAttention ve Dinamik KV Cache.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.paged_attention_motoru import (
    FizikselBlokYonetici,
    GelenIstek,
    PagedKVCache,
    PagedAttentionEngine,
)
from src.fragmentasyon_profilleyici import KVCacheFragmentasyonProfilleyici
from src.gorsellestirici import PagedAttentionGorsellestirici


def test_fiziksel_blok_tahsis_ve_serbest_birakma():
    """1. Fiziksel blok tahsis edildiğinde havuzdan eksilmeli, serbest bırakılınca geri dönmelidir."""
    yonetici = FizikselBlokYonetici(toplam_blok_sayisi=10, blok_boyutu=16)
    b0 = yonetici.blok_tahsis_et()
    assert b0 == 0
    assert len(yonetici.bostaki_bloklar) == 9
    assert yonetici.referans_sayaclari[b0] == 1

    yonetici.blok_serbest_birak(b0)
    assert len(yonetici.bostaki_bloklar) == 10
    assert yonetici.referans_sayaclari[b0] == 0


def test_paged_kv_cache_yazma_ve_okuma():
    """2. PagedKVCache blok boyutu aşıldığında yeni fiziksel blok tahsis etmelidir."""
    yonetici = FizikselBlokYonetici(toplam_blok_sayisi=10, blok_boyutu=16)
    cache = PagedKVCache(toplam_blok_sayisi=10, blok_boyutu=16, num_heads=2, head_dim=32)

    istek = GelenIstek("req_1", prompt_token_sayisi=20)  # 2 Blok gerektirir (16 + 4)
    for _ in range(20):
        k = torch.randn(2, 32)
        v = torch.randn(2, 32)
        cache.token_kv_yaz(istek, k, v, yonetici)

    assert len(istek.blok_tablosu) == 2


def test_paged_attention_kod_cozme_dikkat():
    """3. PagedAttention dağınık fiziksel bloklardan toplanan KV üzerinde dikkat hesaplamalıdır."""
    yonetici = FizikselBlokYonetici(toplam_blok_sayisi=10, blok_boyutu=16)
    cache = PagedKVCache(toplam_blok_sayisi=10, blok_boyutu=16, num_heads=4, head_dim=64)
    engine = PagedAttentionEngine(cache, yonetici)

    istek = GelenIstek("req_2", prompt_token_sayisi=25)
    for _ in range(25):
        k = torch.randn(4, 64)
        v = torch.randn(4, 64)
        cache.token_kv_yaz(istek, k, v, yonetici)

    q = torch.randn(4, 64)
    out = engine.tek_token_dikkat(istek, q)
    assert out.shape == (4, 64)


def test_copy_on_write_referans_sayaci():
    """4. Copy-on-Write (CoW) ile paylaşılan blokların referans sayacı doğru güncellenmelidir."""
    yonetici = FizikselBlokYonetici(toplam_blok_sayisi=10, blok_boyutu=16)
    b0 = yonetici.blok_tahsis_et()
    yonetici.referans_arttir(b0)
    assert yonetici.referans_sayaclari[b0] == 2

    yonetici.blok_serbest_birak(b0)
    assert yonetici.referans_sayaclari[b0] == 1
    assert b0 not in yonetici.bostaki_bloklar


def test_blok_yoneticisi_bellek_tasmasi():
    """5. Blok havuzu tükendiğinde MemoryError fırlatılmalıdır."""
    yonetici = FizikselBlokYonetici(toplam_blok_sayisi=2, blok_boyutu=16)
    yonetici.blok_tahsis_et()
    yonetici.blok_tahsis_et()
    with pytest.raises(MemoryError):
        yonetici.blok_tahsis_et()


def test_fragmentasyon_profilleyici_israf_orani():
    """6. Fragmentasyon profilleyici statik israfın >%70, paged israfın <%10 olduğunu doğrulamalıdır."""
    p = KVCacheFragmentasyonProfilleyici.eszamanli_istek_analizi(istek_sayisi=32)
    assert p["statik_israf_yuzde"] > 70.0
    assert p["paged_israf_yuzde"] < 10.0


def test_eszamanlilik_tarama_raporu():
    """7. Eşzamanlılık raporu 4 farklı istek ölçeğini içermelidir."""
    rapor = KVCacheFragmentasyonProfilleyici.eszamanlilik_tarama_raporu()
    assert len(rapor) == 4
    istekler = [r["istek_sayisi"] for r in rapor]
    assert 128 in istekler


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PagedAttentionGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_paged_attention_paneli.png")
    istek_analizi = KVCacheFragmentasyonProfilleyici.eszamanli_istek_analizi(istek_sayisi=16)
    tarama_raporu = KVCacheFragmentasyonProfilleyici.eszamanlilik_tarama_raporu()

    PagedAttentionGorsellestirici.teshis_paneli_olustur(
        istek_analizi=istek_analizi,
        tarama_raporu=tarama_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
