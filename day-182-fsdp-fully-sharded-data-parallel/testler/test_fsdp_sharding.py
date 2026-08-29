"""
PyTest Birim Testleri - Day 182: Fully Sharded Data Parallel (FSDP).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fsdp_sharding_motoru import ShardingLevel, FSDPKatmanSarmalayici
from src.fsdp_dagitik_yonetici import FSDPModelYoneticisi, FSDPBellekAnalizcisi
from src.gorsellestirici import FSDPGorsellestirici


@pytest.fixture
def ornek_linear_modul():
    """Test için 256x512 doğrusal katman."""
    torch.manual_seed(42)
    return nn.Linear(256, 512, bias=True)


def test_fsdp_katman_sharding_boyutu(ornek_linear_modul):
    """1. FSDPKatmanSarmalayici parametreleri tam olarak 1/N oranında bölmelidir."""
    world_size = 4
    rank = 0
    fsdp_katman = FSDPKatmanSarmalayici(
        module=ornek_linear_modul,
        world_size=world_size,
        rank=rank,
        sharding_level=ShardingLevel.FULL_SHARD,
    )

    toplam_param = 256 * 512 + 512  # 131,584
    assert fsdp_katman.total_param_numel == toplam_param
    assert fsdp_katman.local_shard is not None
    assert fsdp_katman.local_shard.numel() == fsdp_katman.shard_numel
    assert fsdp_katman.shard_numel == pytest.approx(toplam_param / world_size, abs=2)


def test_fsdp_unshard_all_gather_dogruluk():
    """2. All-Gather sonrası elde edilen ağırlık tensörü orijinal ağırlık ile birebir eşleşmelidir."""
    world_size = 4
    ranks_shards = []

    # Orijinal modülü ve ağırlığı al
    torch.manual_seed(42)
    orig_module = nn.Linear(256, 512, bias=True)
    flat_orig = torch.cat([p.data.flatten() for p in orig_module.parameters()])

    # N rank için sarmalayıcıları oluştur
    for r in range(world_size):
        torch.manual_seed(42)
        m = nn.Linear(256, 512, bias=True)
        fsdp = FSDPKatmanSarmalayici(
            module=m,
            world_size=world_size,
            rank=r,
            sharding_level=ShardingLevel.FULL_SHARD,
        )
        ranks_shards.append(fsdp.local_shard.data)

    # Rank 0 üzerinde unshard_parameters çağır
    torch.manual_seed(42)
    m_test = nn.Linear(256, 512, bias=True)
    fsdp_rank0 = FSDPKatmanSarmalayici(
        module=m_test,
        world_size=world_size,
        rank=0,
        sharding_level=ShardingLevel.FULL_SHARD,
    )
    unshared = fsdp_rank0.unshard_parameters(all_rank_shards=ranks_shards)

    assert unshared.shape == flat_orig.shape
    assert torch.allclose(unshared, flat_orig, atol=1e-5)


def test_fsdp_reshard_bellek_bosaltma(ornek_linear_modul):
    """3. reshard_parameters() çağrıldığında unshared ağırlıklar VRAM'den silinmelidir (Drop)."""
    fsdp = FSDPKatmanSarmalayici(
        module=ornek_linear_modul,
        world_size=4,
        rank=0,
        sharding_level=ShardingLevel.FULL_SHARD,
    )

    fsdp.unshard_parameters()
    assert fsdp.unshared_flat_weight is not None

    fsdp.reshard_parameters()
    assert fsdp.unshared_flat_weight is None
    for meta in fsdp.param_metadata:
        assert getattr(fsdp.module, meta["name"], None) is None


def test_fsdp_ileri_gecis_cikti_sekli(ornek_linear_modul):
    """4. FSDP sarmalayıcı üzerinden ileri geçiş doğru çıktıyı üretmelidir."""
    fsdp = FSDPKatmanSarmalayici(
        module=ornek_linear_modul,
        world_size=4,
        rank=0,
        sharding_level=ShardingLevel.FULL_SHARD,
    )

    x = torch.randn(16, 256)
    out = fsdp(x)
    assert out.shape == (16, 512)
    # İleri geçiş bittiğinde ağırlıklar otomatik serbest bırakılmış olmalıdır
    assert fsdp.unshared_flat_weight is None


def test_fsdp_coklu_katman_yonetici():
    """5. FSDPModelYoneticisi çok katmanlı yapıda sıralı icrayı doğru yapmalıdır."""
    layers = [nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)]
    yonetici = FSDPModelYoneticisi(
        layers=layers,
        world_size=4,
        rank=0,
        sharding_level=ShardingLevel.FULL_SHARD,
    )

    rapor = yonetici.get_toplam_bellek_raporu()
    assert rapor["toplam_katman_sayisi"] == 3
    assert rapor["toplam_model_parametre"] > 0
    assert rapor["rank_basina_shard_parametre"] < rapor["toplam_model_parametre"]

    x = torch.randn(8, 64)
    out = yonetici.ileri_gecis(x)
    assert out.shape == (8, 32)


def test_fsdp_bellek_analizcisi_matematiksel_dogruluk():
    """6. FSDPBellekAnalizcisi statik bellek formülleri matematiksel olarak tutarlı olmalıdır."""
    res = FSDPBellekAnalizcisi.statik_bellek_hesapla_gb(param_milyar=70.0, world_size=64, mixed_precision=True)

    assert res["ddp_gb"] == pytest.approx(1043.08, rel=0.1)  # ~1120 GB
    assert res["fsdp_gb"] == pytest.approx(res["ddp_gb"] / 64.0, rel=0.05)
    assert res["fsdp_vram_tasarrufu_yuzde"] > 95.0


def test_fsdp_buyuk_model_karsilastirma_tablosu():
    """7. Büyük modeller tablosu 7B, 13B, 70B ve 175B modelleri içermelidir."""
    tablo = FSDPBellekAnalizcisi.buyuk_model_karsilastirma_tablosu(world_size=64)
    assert len(tablo) == 4
    model_adlari = [m["model_adi"] for m in tablo]
    assert "Llama-2-7B" in model_adlari
    assert "Llama-3-70B" in model_adlari
    assert "GPT-3-175B" in model_adlari


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. FSDPGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti_dosyasi = str(tmp_path / "test_fsdp_paneli.png")
    tablo = FSDPBellekAnalizcisi.buyuk_model_karsilastirma_tablosu(world_size=64)
    layer_stats = [
        {"toplam_parametre_numel": 100000, "shard_parametre_numel": 25000},
        {"toplam_parametre_numel": 200000, "shard_parametre_numel": 50000},
    ]

    FSDPGorsellestirici.fsdp_teshis_paneli_olustur(
        bellek_karsilastirma=tablo,
        layer_stats=layer_stats,
        kayit_yolu=cikti_dosyasi,
    )
    assert os.path.exists(cikti_dosyasi)
    assert os.path.getsize(cikti_dosyasi) > 10000
