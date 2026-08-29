"""
GÜN 144: Tree of Thoughts (ToT) BFS, DFS ve Backtracking Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.dusunce_durumu import DusunceDurumu
from src.durum_degerlendirici import DurumDegerlendirici
from src.tot_arama_motoru import TreeOfThoughtsMotoru
from src.gorsellestirici import TreeOfThoughtsGorsellestirici


def test_dusunce_durumu_baslatma_ve_hedef():
    """DusunceDurumu'nun durum özelliklerini ve hedef kontrolünü doğru yaptığını test eder."""
    durum = DusunceDurumu("node_1", [4, 9, 10, 13])
    assert durum.durum_id == "node_1"
    assert durum.sayilar == [4.0, 9.0, 10.0, 13.0]
    assert durum.hedefe_ulasti_mi(24.0) is False

    hedef_durum = DusunceDurumu("node_final", [24.0])
    assert hedef_durum.hedefe_ulasti_mi(24.0) is True


def test_durum_degerlendirici_kesin_cozum():
    """DurumDegerlendirici'nin [24.0] durumunu kesin çözüm olarak etiketlediğini test eder."""
    durum = DusunceDurumu("node_final", [24.0])
    puan, etiket = DurumDegerlendirici.degerlendir(durum, hedef=24.0)

    assert puan == 1.0
    assert etiket == "kesin_cozum"


def test_durum_degerlendirici_olasi_durum():
    """DurumDegerlendirici'nin [6.0, 4.0] gibi doğrudan 24 üretebilecek durumları olası bulduğunu test eder."""
    durum = DusunceDurumu("node_promise", [6.0, 4.0])
    puan, etiket = DurumDegerlendirici.degerlendir(durum, hedef=24.0)

    assert puan >= 0.90
    assert etiket == "olasi"


def test_durum_degerlendirici_imkansiz_budama():
    """DurumDegerlendirici'nin negatif veya aşırı büyük çıkmazları imkansız etiketlediğini test eder."""
    durum = DusunceDurumu("node_dead", [150.0, 200.0])
    puan, etiket = DurumDegerlendirici.degerlendir(durum, hedef=24.0)

    assert puan <= 0.10
    assert etiket == "imkansiz"


def test_tot_motoru_cocuk_uretme():
    """TreeOfThoughtsMotoru'nun geçerli 4 işlemle çocuk durumlar türettiğini test eder."""
    motor = TreeOfThoughtsMotoru(hedef=24.0)
    kok = DusunceDurumu("root", [4, 9, 10, 13])
    cocuklar = motor._cocuk_durumlari_uret(kok)

    assert len(cocuklar) > 0
    assert all(len(c.sayilar) == 3 for c in cocuklar)
    assert all(len(c.adim_gecmisi) == 1 for c in cocuklar)


def test_tot_bfs_arama_24_oyunu():
    """Tree of Thoughts BFS algoritmasının [4, 9, 10, 13] kümesinden 24'e ulaştığını test eder."""
    motor = TreeOfThoughtsMotoru(hedef=24.0)
    sonuc = motor.bfs_ara([4, 9, 10, 13], beam_genisligi=4)

    assert sonuc["cozum_bulundu_mu"] is True
    assert sonuc["nihai_sayi"] == 24.0
    assert len(sonuc["adim_gecmisi"]) == 3
    assert sonuc["toplam_budanan_dugum"] > 0


def test_tot_dfs_arama_backtracking():
    """Tree of Thoughts DFS algoritmasının geri izleme ile 24 çözümüne ulaştığını test eder."""
    motor = TreeOfThoughtsMotoru(hedef=24.0)
    sonuc = motor.dfs_ara([4, 9, 10, 13], maks_derinlik=4)

    assert sonuc["cozum_bulundu_mu"] is True
    assert sonuc["nihai_sayi"] == 24.0
    assert "Tree of Thoughts (DFS + Backtracking)" in sonuc["algoritma"]


def test_gorsellestirici_pano_uretme():
    """6 panelli ToT teşhis panosunun PNG olarak başarıyla kaydedildiğini test eder."""
    motor = TreeOfThoughtsMotoru(hedef=24.0)
    bfs_res = motor.bfs_ara([4, 9, 10, 13], beam_genisligi=3)
    dfs_res = motor.dfs_ara([4, 9, 10, 13], maks_derinlik=4)

    gorsellestirici = TreeOfThoughtsGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_tot_pano.png")
        gorsellestirici.pano_olustur(bfs_res, dfs_res, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
