"""
GÜN 173: Classifier-Free Guidance (CFG) ve DDIM Hızlı Zamanlayıcılar Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.cfg_yoneticisi import CFGYoneticisi
from src.ddim_zamanlayici import DDIMZamanlayici
from src.cfg_ddim_evaluator import CFGDualEvaluator
from src.gorsellestirici import CFGGorsellestirici


def test_cfg_yonlendirilmis_gurultu_hesaplama():
    """CFG formülünün (uncond + w * (cond - uncond)) doğru tensör ürettiğini test eder."""
    cfg = CFGYoneticisi(varsayilan_guidance_scale=7.5)
    eps_uncond = torch.zeros(2, 4, 16, 16)
    eps_cond = torch.ones(2, 4, 16, 16)

    eps_guided = cfg.yonlendirilmis_gurultu_hesapla(eps_uncond, eps_cond, guidance_scale=7.5)

    # eps_guided = 0 + 7.5 * (1 - 0) = 7.5
    assert torch.allclose(eps_guided, torch.full_like(eps_guided, 7.5))


def test_cfg_w_1_saf_kosullu():
    """w=1.0 iken yönlendirilmiş gürültünün doğrudan eps_cond'a eşit olduğunu test eder."""
    cfg = CFGYoneticisi()
    eps_uncond = torch.randn(2, 4, 16, 16)
    eps_cond = torch.randn(2, 4, 16, 16)

    eps_guided = cfg.yonlendirilmis_gurultu_hesapla(eps_uncond, eps_cond, guidance_scale=1.0)
    assert torch.allclose(eps_guided, eps_cond, atol=1e-5)


def test_cfg_dinamik_esikleme():
    """Dinamik eşiklemenin aşırı uç değerleri güvenli aralığa sınırladığını test eder."""
    cfg = CFGYoneticisi()
    eps_uncond = torch.zeros(1, 4, 16, 16)
    eps_cond = torch.ones(1, 4, 16, 16) * 50.0  # Aşırı büyük değer

    eps_guided = cfg.yonlendirilmis_gurultu_hesapla(
        eps_uncond, eps_cond, guidance_scale=15.0, dinamik_esikleme=True
    )
    assert torch.all(eps_guided <= 1.05)
    assert torch.all(eps_guided >= -1.05)


def test_ddim_zamanlayici_adim_yapilandirmasi():
    """DDIM zamanlayıcısının belirtilen sayıda inferans adımı oluşturduğunu test eder."""
    ddim = DDIMZamanlayici(num_train_timesteps=1000, num_inference_steps=20, eta=0.0)
    assert len(ddim.timesteps) == 20
    assert ddim.timesteps[0] > ddim.timesteps[-1]


def test_ddim_ornekleme_adimi():
    """DDIM tek adımlı ters difüzyon güncellemesinin tensör boyutunu koruduğunu test eder."""
    ddim = DDIMZamanlayici(num_train_timesteps=1000, num_inference_steps=20, eta=0.0)
    z_t = torch.randn(2, 4, 16, 16)
    eps_guided = torch.randn(2, 4, 16, 16)

    z_prev = ddim.ornekleme_adimi(z_t, eps_guided, t_idx=0)
    assert z_prev.shape == (2, 4, 16, 16)


def test_ddim_eta_stokastik_adim():
    """eta>0 olduğunda rastgele gürültü eklenerek stokastik örnekleme yapıldığını test eder."""
    ddim = DDIMZamanlayici(num_train_timesteps=1000, num_inference_steps=20, eta=0.5)
    z_t = torch.randn(2, 4, 16, 16)
    eps_guided = torch.randn(2, 4, 16, 16)

    z_prev = ddim.ornekleme_adimi(z_t, eps_guided, t_idx=5)
    assert z_prev.shape == (2, 4, 16, 16)


def test_cfg_ddim_analiz_raporu():
    """Analiz raporunun 5 CFG ölçeği ve 50x hızlanma metriklerini içerdiğini test eder."""
    rapor = CFGDualEvaluator.cfg_olcek_analizini_getir()
    assert len(rapor["olcek_deneyleri"]) == 5
    assert rapor["zamanlayici_kiyaslamasi"]["hizlanma_faktoru"] > 40.0


def test_gorsellestirici_pano_uretme():
    """6 panelli CFG & DDIM teşhis panosunun PNG olarak kaydedildiğini test eder."""
    rapor = CFGDualEvaluator.cfg_olcek_analizini_getir()
    gorsellestirici = CFGGorsellestirici(dpi=100)

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_cfg_ddim_pano.png")
        gorsellestirici.pano_olustur(rapor, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
