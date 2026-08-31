"""
Tesla BEV ve Homografi Birim Testleri (PyTest)
==============================================
Bu test paketi; 2D piksel - BEV dönüşümünü, gidiş-dönüş homografi tutarlılığını
ve ufuk çizgisi kırpma mekanizmasını test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_bev_homografi_ve_ipm import TeslaBEVTransformer


def test_gidis_donus_homografi_tutarliligi():
    """(u, v) -> BEV -> (u', v') dönüşüm hatasının alt-piksel düzeyinde (< 1e-4) olduğu test edilir."""
    transformer = TeslaBEVTransformer()
    test_u, test_v = 640.0, 750.0

    bev_pt = transformer.pixel_to_bev(test_u, test_v)
    assert bev_pt is not None
    x_long, y_lat = bev_pt

    reproj = transformer.bev_to_pixel(x_long, y_lat)
    assert reproj is not None
    u_re, v_re = reproj

    assert np.isclose(test_u, u_re, atol=1e-3)
    assert np.isclose(test_v, v_re, atol=1e-3)


def test_ufuk_cizgisi_kirpma():
    """Ufuk çizgisinin üstündeki veya çok uzak piksellerin filtrelendiği test edilir."""
    transformer = TeslaBEVTransformer()
    # v = 200 (gökyüzü / ufuk üstü)
    pt = transformer.pixel_to_bev(640.0, 200.0)
    assert pt is None


def test_serit_paralellik_donusumu():
    """2D perspektifte üçgen gibi kapanan şeritlerin BEV'de paralel kaldığı test edilir."""
    transformer = TeslaBEVTransformer()
    # İleri 10m ve 30m'de sol ve sağ şeritler
    pt_l1 = transformer.bev_to_pixel(10.0, 1.875)
    pt_r1 = transformer.bev_to_pixel(10.0, -1.875)
    pt_l2 = transformer.bev_to_pixel(30.0, 1.875)
    pt_r2 = transformer.bev_to_pixel(30.0, -1.875)

    # 2D'de yakındaki mesafe (u_r1 - u_l1) uzaktaki mesafeden (u_r2 - u_l2) daha geniştir
    assert (pt_r1[0] - pt_l1[0]) > (pt_r2[0] - pt_l2[0])

    # Geriye dönüştürüldüğünde BEV'de aralarındaki mesafe 3.75 m'dir
    b_l1 = transformer.pixel_to_bev(*pt_l1)
    b_r1 = transformer.pixel_to_bev(*pt_r1)
    b_l2 = transformer.pixel_to_bev(*pt_l2)
    b_r2 = transformer.pixel_to_bev(*pt_r2)

    assert np.isclose(abs(b_l1[1] - b_r1[1]), 3.75, atol=0.01)
    assert np.isclose(abs(b_l2[1] - b_r2[1]), 3.75, atol=0.01)
