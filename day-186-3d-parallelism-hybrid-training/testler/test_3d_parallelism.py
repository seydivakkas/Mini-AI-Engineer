"""
PyTest Birim Testleri - Day 186: 3D Paralellik (DP + TP + PP).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.uc_boyutlu_grid_topolojisi import UcBoyutluGridTopolojisi
from src.hibrit_3d_egitim_motoru import Hibrit3DEgitimMotoru
from src.gorsellestirici import UcBoyutluGorsellestirici


@pytest.fixture
def ornek_grid():
    """Test için DP=2, PP=4, TP=8 (Toplam 64 GPU) 3D topoloji."""
    return UcBoyutluGridTopolojisi(dp_size=2, pp_size=4, tp_size=8)


def test_uc_boyutlu_grid_olusturma(ornek_grid):
    """1. 3D grid toplam GPU sayısını (world_size) doğru hesaplamalıdır."""
    assert ornek_grid.world_size == 64
    assert ornek_grid.dp_size == 2
    assert ornek_grid.pp_size == 4
    assert ornek_grid.tp_size == 8


def test_rank_koordinat_cift_yonlu_donusum(ornek_grid):
    """2. Rank <-> (dp, pp, tp) koordinat dönüşümü birebir (bisection) olmalıdır."""
    for r in range(ornek_grid.world_size):
        dp_r, pp_r, tp_r = ornek_grid.get_coordinates(r)
        r_ters = ornek_grid.get_rank(dp_r, pp_r, tp_r)
        assert r == r_ters


def test_tp_grup_boyutu_ve_elemanlari(ornek_grid):
    """3. TP grubu tam olarak TP_size elemandan oluşmalı ve aynı DP, PP koordinatını paylaşmalıdır."""
    tp_grp = ornek_grid.get_tp_group(rank=27)
    assert len(tp_grp) == 8
    dp_ilk, pp_ilk, _ = ornek_grid.get_coordinates(tp_grp[0])
    for r in tp_grp:
        dp_r, pp_r, _ = ornek_grid.get_coordinates(r)
        assert dp_r == dp_ilk
        assert pp_r == pp_ilk


def test_pp_grup_boyutu_ve_elemanlari(ornek_grid):
    """4. PP grubu tam olarak PP_size elemandan oluşmalı ve aynı DP, TP koordinatını paylaşmalıdır."""
    pp_grp = ornek_grid.get_pp_group(rank=27)
    assert len(pp_grp) == 4
    dp_ilk, _, tp_ilk = ornek_grid.get_coordinates(pp_grp[0])
    for r in pp_grp:
        dp_r, _, tp_r = ornek_grid.get_coordinates(r)
        assert dp_r == dp_ilk
        assert tp_r == tp_ilk


def test_dp_grup_boyutu_ve_elemanlari(ornek_grid):
    """5. DP grubu tam olarak DP_size elemandan oluşmalı ve aynı PP, TP koordinatını paylaşmalıdır."""
    dp_grp = ornek_grid.get_dp_group(rank=27)
    assert len(dp_grp) == 2
    _, pp_ilk, tp_ilk = ornek_grid.get_coordinates(dp_grp[0])
    for r in dp_grp:
        _, pp_r, tp_r = ornek_grid.get_coordinates(r)
        assert pp_r == pp_ilk
        assert tp_r == tp_ilk


def test_hibrit_3d_vram_profili_llama70b():
    """6. Llama-3-70B modeli 64 GPU'lu 3D gridde 80GB H100 VRAM sınırının çok altında kalmalıdır."""
    profil = Hibrit3DEgitimMotoru.vram_ve_kaynak_profili(
        model_adi="Llama-3-70B",
        dp_size=2,
        pp_size=4,
        tp_size=8,
        zero_dp_etkin=True,
    )
    assert profil["vram_sigiyor_mu"] is True
    assert profil["gpu_toplam_vram_gb"] < 40.0


def test_tum_modeller_raporu_gecerlilik():
    """7. Analiz edilen tüm modeller (70B, 175B, 405B) önerilen 3D grid konfigürasyonunda VRAM sınırlarına sığmalıdır."""
    rapor = Hibrit3DEgitimMotoru.tum_modeller_analiz_raporu()
    assert len(rapor) == 3
    for r in rapor:
        assert r["vram_sigiyor_mu"] is True
        assert r["mfu_yuzde"] > 50.0


def test_gorsellestirme_cikti_dosyasi(tmp_path):
    """8. UcBoyutluGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_3d_paneli.png")
    rapor = Hibrit3DEgitimMotoru.tum_modeller_analiz_raporu()

    UcBoyutluGorsellestirici.teshis_paneli_olustur(
        model_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
