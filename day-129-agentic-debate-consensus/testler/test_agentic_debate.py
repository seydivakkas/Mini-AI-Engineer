"""
FAZ 7 GÜN 129: Multi-Agent Debate & Consensus Voting Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.tartismaci_ajanlar import MuhafazakarAjan, YenilikciAjan, PragmatikAjan
from src.hakem_ve_oylama import HakemAjan, KonsensusOylayici
from src.tartisma_motoru import MultiAgentTartismaMotoru
from src.gorsellestirici import DebateGorsellestirici


def test_muhafazakar_ajan_arguman_uret():
    """MuhafazakarAjan'ın güvenlik ve sıfır güven odaklı tez ürettiğini test eder."""
    ajan = MuhafazakarAjan()
    arg = ajan.arguman_uret("Mikroservis Güvenlik Mimarisi", tur_no=1, diger_argumanlar=[])

    assert arg["tur"] == 1
    assert "Guvenlik" in arg["tercih_edilen_secenek"]
    assert arg["guven_skoru"] >= 0.85
    assert len(arg["vurgulanan_riskler"]) >= 2


def test_yenilikci_ajan_arguman_uret():
    """YenilikciAjan'ın hız, ölçek ve düşük gecikme odaklı tez ürettiğini test eder."""
    ajan = YenilikciAjan()
    arg = ajan.arguman_uret("Mikroservis Güvenlik Mimarisi", tur_no=1, diger_argumanlar=[])

    assert arg["tur"] == 1
    assert "Performans" in arg["tercih_edilen_secenek"]
    assert arg["guven_skoru"] >= 0.85


def test_pragmatik_ajan_arguman_uret():
    """PragmatikAjan'ın maliyet ve hibrit denge odaklı tez ürettiğini test eder."""
    ajan = PragmatikAjan()
    arg = ajan.arguman_uret("Mikroservis Güvenlik Mimarisi", tur_no=3, diger_argumanlar=[])

    assert arg["tur"] == 3
    assert "Hibrit" in arg["tercih_edilen_secenek"]
    assert arg["guven_skoru"] >= 0.90


def test_hakem_ajan_tur_degerlendir():
    """HakemAjan'ın argümanları puanlayıp konsensüs durumunu saptadığını test eder."""
    hakem = HakemAjan()
    argumanlar = [
        {"ajan": "Ajan A", "guven_skoru": 0.9, "tez": "Güvenlik şart"},
        {"ajan": "Ajan B", "guven_skoru": 0.85, "tez": "Hız şart"},
    ]

    rapor = hakem.tur_degerlendir(tur_no=1, argumanlar=argumanlar)
    assert rapor["tur_no"] == 1
    assert "Ajan A" in rapor["ajan_skorlari"]
    assert rapor["ajan_skorlari"]["Ajan A"] > 70.0


def test_konsensus_oylayici_cogunluk_ve_agirlikli():
    """KonsensusOylayici'nın hem çoğunluk hem de ağırlıklı güven oylamasını doğru hesapladığını test eder."""
    oylayici = KonsensusOylayici()
    argumanlar = [
        {"tercih_edilen_secenek": "Secenek_X", "guven_skoru": 0.95},
        {"tercih_edilen_secenek": "Secenek_X", "guven_skoru": 0.90},
        {"tercih_edilen_secenek": "Secenek_Y", "guven_skoru": 0.70},
    ]

    cogunluk = oylayici.cogunluk_oylamasi(argumanlar)
    assert cogunluk["kazanan_secenek"] == "Secenek_X"
    assert cogunluk["kazanan_oy_orani"] > 60.0

    agirlikli = oylayici.agirlikli_guven_oylamasi(argumanlar)
    assert agirlikli["kazanan_secenek"] == "Secenek_X"
    assert agirlikli["kazanan_guven_orani"] > 50.0


def test_multi_agent_tartisma_motoru_tam_akis():
    """MultiAgentTartismaMotoru'nun çok turlu tartışmayı ve hakem hükmünü eksiksiz tamamladığını test eder."""
    motor = MultiAgentTartismaMotoru(max_tur=3)
    sonuc = motor.tartismayi_yurut("Ödeme Altyapısı Mimarisi")

    assert sonuc["toplam_tur"] >= 2
    assert len(sonuc["tur_kayitlari"]) >= 2
    assert sonuc["nihai_hukum"]["karar_kesin_mi"] is True
    assert "HAKEM" in sonuc["nihai_hukum"]["hukum_metni"]


def test_benchmark_karsilastir_metrikleri():
    """MultiAgentTartismaMotoru kıyaslama metriklerinin eksiksiz olduğunu test eder."""
    motor = MultiAgentTartismaMotoru()
    bench = motor.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["multi_agent_debate"][0] > bench["tekil_model"][0]


def test_debate_gorsellestirici_pano():
    """DebateGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    motor = MultiAgentTartismaMotoru(max_tur=3)
    rapor = motor.tartismayi_yurut("Mikroservis vs Monolit")
    bench = motor.benchmark_karsilastir()

    gorsellestirici = DebateGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_debate_pano.png")
        gorsellestirici.pano_olustur(rapor, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
