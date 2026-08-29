"""
GÜN 146: Monte Carlo Tree Search (MCTS) Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.mcts_dugumu import MCTSDugumu
from src.rollout_politika_motoru import RolloutPolitikaMotoru
from src.mcts_planlayici import MCTSDusuncePlanlayici
from src.gorsellestirici import MCTSGorsellestirici


def test_mcts_dugumu_baslatma():
    """MCTSDugumu'nun doğru başlatıldığını ve sıfır ziyaret değerlerini test eder."""
    dugum = MCTSDugumu("root", [4, 9, 10, 13])
    assert dugum.durum_id == "root"
    assert dugum.ziyaret_sayisi == 0
    assert dugum.toplam_odul == 0.0
    assert dugum.ortalama_deger == 0.0
    assert dugum.terminal_mi() is False


def test_uct_skoru_hesaplama():
    """UCT skorunun keşfedilmemiş düğümlere sonsuz, keşfedilenlere dengeli skor verdiğini test eder."""
    kok = MCTSDugumu("root", [4, 9, 10, 13])
    kok.ziyaret_sayisi = 10

    cocuk_yeni = MCTSDugumu("c1", [13 - 9, 4, 10], ebeveyn=kok)
    assert cocuk_yeni.uct_skoru() == float("inf")

    cocuk_ziyaretli = MCTSDugumu("c2", [10 - 4, 9, 13], ebeveyn=kok)
    cocuk_ziyaretli.ziyaret_sayisi = 4
    cocuk_ziyaretli.toplam_odul = 3.2  # Q = 0.8
    assert cocuk_ziyaretli.uct_skoru() > 0.80


def test_rollout_politika_terminal_dogrulama():
    """RolloutPolitikaMotoru'nun 24 hedefine ulaşıldığında 1.0 döndürdüğünü test eder."""
    odul = RolloutPolitikaMotoru.simule_et([24.0], hedef=24.0)
    assert odul == 1.0

    odul_iki_sayi = RolloutPolitikaMotoru.simule_et([6.0, 4.0], hedef=24.0)
    assert odul_iki_sayi == 1.0


def test_rollout_politika_imkansiz_durum():
    """RolloutPolitikaMotoru'nun imkansız veya aşırı büyük durumlarda 0.0 döndürdüğünü test eder."""
    odul = RolloutPolitikaMotoru.simule_et([150.0, 200.0], hedef=24.0)
    assert odul == 0.0


def test_mcts_planlayici_cocuk_turetme():
    """MCTSDusuncePlanlayici'nin geçerli 4 işlem kombinasyonlarıyla çocuk türettiğini test eder."""
    planlayici = MCTSDusuncePlanlayici(hedef=24.0)
    kok = MCTSDugumu("root", [4, 9, 10, 13])
    cocuklar = planlayici._cocuk_dugumleri_turet(kok)

    assert len(cocuklar) > 0
    assert all(len(c.sayilar) == 3 for c in cocuklar)
    assert all(c.ebeveyn == kok for c in cocuklar)


def test_mcts_planlayici_24_oyunu_cozumu():
    """MCTS algoritmasının [4, 9, 10, 13] kümesinden 24 hedefine ulaştığını test eder."""
    planlayici = MCTSDusuncePlanlayici(hedef=24.0, simulasyon_sayisi=70)
    sonuc = planlayici.planla([4, 9, 10, 13])

    assert sonuc["cozum_bulundu_mu"] is True
    assert sonuc["nihai_sayi"] == 24.0
    assert len(sonuc["en_iyi_yol_adimlari"]) == 3
    assert sonuc["kok_ziyaret_sayisi"] == 70


def test_mcts_geri_yayilim_guncelleme():
    """Geri yayılım adımının köke kadar ziyaret sayısı ve toplam ödülü artırdığını test eder."""
    kok = MCTSDugumu("root", [4, 9, 10, 13])
    cocuk = MCTSDugumu("c1", [4, 4, 10], ebeveyn=kok)
    torun = MCTSDugumu("t1", [6, 4], ebeveyn=cocuk)

    odul = 1.0
    gezen = torun
    while gezen is not None:
        gezen.ziyaret_sayisi += 1
        gezen.toplam_odul += odul
        gezen = gezen.ebeveyn

    assert kok.ziyaret_sayisi == 1
    assert kok.toplam_odul == 1.0
    assert cocuk.ziyaret_sayisi == 1
    assert torun.ziyaret_sayisi == 1


def test_gorsellestirici_pano_uretme():
    """6 panelli MCTS teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    planlayici = MCTSDusuncePlanlayici(hedef=24.0, simulasyon_sayisi=20)
    sonuc = planlayici.planla([4, 9, 10, 13])

    gorsellestirici = MCTSGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_mcts_pano.png")
        gorsellestirici.pano_olustur(sonuc, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
