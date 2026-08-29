"""
PyTest Birim Testleri - Day 192: Continuous Batching ve Chunked Prefill.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.continuous_batching_motoru import (
    IstekDurumu,
    LLMIstek,
    ContinuousBatchingScheduler,
)
from src.kuyruk_gecikme_profilleyici import KuyrukGecikmeProfilleyici
from src.gorsellestirici import ContinuousBatchingGorsellestirici


def test_llm_istek_olusturma_ve_metrikler():
    """1. LLMIstek oluşturulmalı ve TTFT/TPOT metrikleri doğru hesaplanmalıdır."""
    istek = LLMIstek("req_1", varis_zamani=1.0, prompt_token_sayisi=100, hedef_uretim_token=10)
    assert istek.durum == IstekDurumu.BEKLEMEDE
    assert istek.ttft is None

    istek.ilk_token_zamani = 1.5
    assert istek.ttft == pytest.approx(0.5, abs=1e-3)

    istek.uretilen_token_sayisi = 10
    istek.bitis_zamani = 2.4
    assert istek.tpot == pytest.approx((2.4 - 1.5) / 9.0, abs=1e-3)


def test_zamanlayici_kuyruga_istek_ekleme():
    """2. Zamanlayıcıya eklenen istek kuyruk listesine girmelidir."""
    scheduler = ContinuousBatchingScheduler(max_batch_size=4)
    istek = LLMIstek("req_2", varis_zamani=0.0, prompt_token_sayisi=50, hedef_uretim_token=5)
    scheduler.istek_ekle(istek)
    assert len(scheduler.kuyruktaki_istekler) == 1


def test_chunked_prefill_dilimleme():
    """3. Chunked Prefill uzun promptları chunk_size dilimlerine bölerek işlemelidir."""
    scheduler = ContinuousBatchingScheduler(max_batch_size=2, max_batched_tokens=256, chunk_size=100)
    istek = LLMIstek("req_long", varis_zamani=0.0, prompt_token_sayisi=250, hedef_uretim_token=5)
    scheduler.istek_ekle(istek)

    # Adım 1: İlk 100 token
    scheduler.adim_yurut(0.1)
    assert istek.islenen_prompt_token == 100
    assert istek.durum == IstekDurumu.PREFILL

    # Adım 2: İkinci 100 token
    scheduler.adim_yurut(0.2)
    assert istek.islenen_prompt_token == 200
    assert istek.durum == IstekDurumu.PREFILL

    # Adım 3: Son 50 token -> DECODE durumuna geçmeli
    scheduler.adim_yurut(0.3)
    assert istek.islenen_prompt_token == 250
    assert istek.durum == IstekDurumu.DECODE


def test_aninda_tahliye_eviction():
    """4. Hedef token sayısına ulaşan istekler anında tamamlananlar havuzuna aktarılmalıdır."""
    scheduler = ContinuousBatchingScheduler(max_batch_size=2, max_batched_tokens=256, chunk_size=100)
    istek = LLMIstek("req_quick", varis_zamani=0.0, prompt_token_sayisi=50, hedef_uretim_token=2)
    scheduler.istek_ekle(istek)

    # Prefill adımı
    scheduler.adim_yurut(0.1)
    assert istek.durum == IstekDurumu.DECODE

    # Decode Adım 1
    scheduler.adim_yurut(0.2)
    assert istek.uretilen_token_sayisi == 1

    # Decode Adım 2 -> Tamamlandı
    scheduler.adim_yurut(0.3)
    assert istek.uretilen_token_sayisi == 2
    assert istek.durum == IstekDurumu.TAMAMLANDI
    assert len(scheduler.tamamlanan_istekler) == 1
    assert len(scheduler.calisan_istekler) == 0


def test_prefill_decode_harmanlama():
    """5. Prefill ve Decode istekleri aynı adımda bütçe dahilinde harmanlanmalıdır."""
    scheduler = ContinuousBatchingScheduler(max_batch_size=4, max_batched_tokens=512, chunk_size=128)
    req_dec = LLMIstek("req_dec", varis_zamani=0.0, prompt_token_sayisi=10, hedef_uretim_token=10)
    req_pref = LLMIstek("req_pref", varis_zamani=0.0, prompt_token_sayisi=100, hedef_uretim_token=5)

    scheduler.istek_ekle(req_dec)
    scheduler.adim_yurut(0.1)  # req_dec decode'a geçer
    scheduler.istek_ekle(req_pref)

    t = scheduler.adim_yurut(0.2)
    assert t["decode_istek_sayisi"] == 1
    assert req_pref.islenen_prompt_token == 100


def test_kuyruk_gecikme_profilleyici_ttft_iyilesmesi():
    """6. Gecikme profilleyicisi Continuous Batching ile TTFT iyileşmesini doğrulamalıdır."""
    sonuclar = KuyrukGecikmeProfilleyici.karsilastirmali_simulasyon_yurut(toplam_istek=15)
    assert sonuclar["cb_ortalama_ttft_sn"] < sonuclar["statik_ortalama_ttft_sn"]


def test_kuyruk_gecikme_profilleyici_toplam_sure():
    """7. Toplam işleme süresi Continuous Batching'de belirgin şekilde daha düşük olmalıdır."""
    sonuclar = KuyrukGecikmeProfilleyici.karsilastirmali_simulasyon_yurut(toplam_istek=15)
    assert sonuclar["cb_toplam_sure_sn"] < sonuclar["statik_toplam_sure_sn"]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ContinuousBatchingGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_cb_paneli.png")
    sonuclar = KuyrukGecikmeProfilleyici.karsilastirmali_simulasyon_yurut(toplam_istek=10)

    ContinuousBatchingGorsellestirici.teshis_paneli_olustur(
        simulasyon_sonuclari=sonuclar,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
