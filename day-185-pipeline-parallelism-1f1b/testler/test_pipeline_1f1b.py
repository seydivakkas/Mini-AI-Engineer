"""
PyTest Birim Testleri - Day 185: Pipeline Parallelism (PP) & 1F1B.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline_paralellik_motoru import PipelineStage, P2PIletisimKuyrugu
from src.zaman_cizelgesi_1f1b import ZamanCizelgesiTuru, PipelineZamanCizelgesiMotoru
from src.gorsellestirici import PipelineGorsellestirici


@pytest.fixture
def ornek_stage():
    """Test için 2 katmanlı pipeline aşaması."""
    layers = nn.ModuleList([nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)])
    return PipelineStage(layers=layers, stage_id=0, num_stages=4)


def test_pipeline_stage_forward_ve_cache(ornek_stage):
    """1. PipelineStage ileri geçişte aktivasyonu önbelleğe almalı ve doğru çıktıyı üretmelidir."""
    x = torch.randn(4, 64)
    out = ornek_stage.forward_step(microbatch_id=0, input_tensor=x)
    assert out.shape == (4, 64)
    assert ornek_stage.get_cached_activation_count() == 1


def test_pipeline_stage_backward_ve_grad(ornek_stage):
    """2. backward_step() girişe göre gradyan üretmeli ve önbelleği temizlemelidir."""
    x = torch.randn(4, 64)
    out = ornek_stage.forward_step(microbatch_id=0, input_tensor=x)

    grad_out = torch.ones_like(out)
    in_grad = ornek_stage.backward_step(microbatch_id=0, output_grad=grad_out)

    assert in_grad.shape == (4, 64)
    assert ornek_stage.get_cached_activation_count() == 0


def test_p2p_iletisim_kuyrugu_ileri_ve_geri():
    """3. P2PIletisimKuyrugu aşamalar arası forward ve backward transferlerini doğru yapmalıdır."""
    kuyruk = P2PIletisimKuyrugu(num_stages=4)
    t_fwd = torch.randn(2, 32)
    kuyruk.forward_gonder(from_stage=0, microbatch_id=1, tensor=t_fwd)

    rec_fwd = kuyruk.forward_al(to_stage=1, microbatch_id=1)
    assert rec_fwd is not None
    assert torch.allclose(rec_fwd, t_fwd)

    t_bwd = torch.randn(2, 32)
    kuyruk.backward_gonder(from_stage=1, microbatch_id=1, grad_tensor=t_bwd)
    rec_bwd = kuyruk.backward_al(to_stage=0, microbatch_id=1)
    assert rec_bwd is not None
    assert torch.allclose(rec_bwd, t_bwd)


def test_balon_orani_matematiksel_dogruluk():
    """4. Balon oranı formülü (P-1)/(M+P-1) matematiksel olarak tutarlı olmalıdır."""
    # P=8, M=32 -> (8-1)/(32+8-1) = 7 / 39 = 0.17948 (%17.9)
    b = PipelineZamanCizelgesiMotoru.balon_orani_hesapla(num_stages=8, num_microbatches=32, virtual_stages=1)
    assert b == pytest.approx(7.0 / 39.0, abs=1e-4)


def test_interleaved_balon_azaltma():
    """5. Sanal aşamalar (v=2) balon oranını standart 1F1B'ye göre azaltmalıdır."""
    b_std = PipelineZamanCizelgesiMotoru.balon_orani_hesapla(num_stages=8, num_microbatches=32, virtual_stages=1)
    b_int = PipelineZamanCizelgesiMotoru.balon_orani_hesapla(num_stages=8, num_microbatches=32, virtual_stages=2)
    assert b_int < b_std
    assert b_int == pytest.approx(7.0 / 64.0, abs=1e-4)


def test_tepe_aktivasyon_bellegi_gpipe_vs_1f1b():
    """6. 1F1B tepe aktivasyon belleği O(P) olmalı ve GPipe'ın O(M) belleğinden belirgin düşük olmalıdır."""
    p_stages = 8
    m_batches = 32
    act_mb = 100.0

    gpipe_mb = PipelineZamanCizelgesiMotoru.tepe_aktivasyon_bellegi_mb(
        cizelge_turu=ZamanCizelgesiTuru.NAIVE_GPIPE,
        num_stages=p_stages,
        num_microbatches=m_batches,
        microbatch_aktivasyon_mb=act_mb,
    )
    f1b_mb = PipelineZamanCizelgesiMotoru.tepe_aktivasyon_bellegi_mb(
        cizelge_turu=ZamanCizelgesiTuru.SCHEDULE_1F1B,
        num_stages=p_stages,
        num_microbatches=m_batches,
        microbatch_aktivasyon_mb=act_mb,
    )

    assert gpipe_mb == 3200.0  # 32 * 100
    assert f1b_mb == 800.0     # 8 * 100
    assert f1b_mb < gpipe_mb


def test_karsilastirmali_cizelge_raporu():
    """7. Karşılaştırmalı çizelge raporu GPipe, 1F1B ve Interleaved 1F1B olmak üzere 3 kaydı içermelidir."""
    rapor = PipelineZamanCizelgesiMotoru.karsilastirmali_cizelge_raporu(num_stages=8, num_microbatches=32)
    assert len(rapor) == 3
    adlar = [r["cizelge_adi"] for r in rapor]
    assert any("GPipe" in a for a in adlar)
    assert any("1F1B" in a for a in adlar)


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. PipelineGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_pp_paneli.png")
    rapor = PipelineZamanCizelgesiMotoru.karsilastirmali_cizelge_raporu(num_stages=8, num_microbatches=32)

    PipelineGorsellestirici.pipeline_teshis_paneli_olustur(
        cizelge_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
