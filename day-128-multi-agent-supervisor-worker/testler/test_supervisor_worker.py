"""
FAZ 7 GÜN 128: Multi-Agent Supervisor-Worker Test Paketi.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.ajan_rolleri import ArastirmaciAjan, GelistiriciAjan, DenetleyiciAjan
from src.supervisor_yonetici import SupervisorAjan
from src.gorsellestirici import MultiAgentGorsellestirici


def test_arastirmaci_ajan_gorev_yap():
    """ArastirmaciAjan'ın problem analizini ve algoritma kısıtlarını başarıyla çıkardığını test eder."""
    arastirmaci = ArastirmaciAjan()
    sonuc = arastirmaci.gorev_yap({"problem": "Maksimum alt dizi toplamı bulma"})

    assert sonuc["arastirma_tamamlandi"] is True
    assert "Kadane" in sonuc["secilen_algoritma"]
    assert sonuc["zaman_karmasikligi"] == "O(N)"
    assert len(sonuc["onemli_kisitlar"]) >= 3


def test_gelistirici_ajan_ilk_surum_ve_revizyon():
    """GelistiriciAjan'ın ilk taslak kodu ve denetçi eleştirisi sonrası revize kodu ürettiğini test eder."""
    gelistirici = GelistiriciAjan()

    # 1. Tur Taslak
    v1 = gelistirici.gorev_yap({"problem": "Max Subarray"})
    assert v1["versiyon"] == 1
    assert "max_alt_dizi" in v1["kod"]

    # 2. Tur Revizyon
    v2 = gelistirici.gorev_yap({"problem": "Max Subarray", "denetci_elestirisi": "Negatif sayılarda hata var"})
    assert v2["versiyon"] == 2
    assert "dizi[0]" in v2["kod"]


def test_denetleyici_ajan_kusurlu_kod_tespiti():
    """DenetleyiciAjan'ın hatalı taslak kodu reddedip yapıcı eleştiri sunduğunu test eder."""
    denetleyici = DenetleyiciAjan()
    kusurlu_kod = "def max_alt_dizi(dizi):\n    max_toplam = 0\n    return max_toplam"

    sonuc = denetleyici.gorev_yap({"kod": kusurlu_kod})
    assert sonuc["onaylandi"] is False
    assert sonuc["kalite_skoru"] < 70.0
    assert "HATA TESPİTİ" in sonuc["elestiri"]


def test_denetleyici_ajan_kusursuz_kod_onayi():
    """DenetleyiciAjan'ın doğru ve optimize kodu onayladığını test eder."""
    denetleyici = DenetleyiciAjan()
    kusursuz_kod = "def max_alt_dizi(dizi):\n    mevcut = max_t = dizi[0]\n    return max_t"

    sonuc = denetleyici.gorev_yap({"kod": kusursuz_kod})
    assert sonuc["onaylandi"] is True
    assert sonuc["kalite_skoru"] > 90.0
    assert "ONAYLANDI" in sonuc["elestiri"]


def test_supervisor_ajan_orkestrasyon_akis():
    """SupervisorAjan'ın işçileri doğru sırayla çağırıp görevi başarıyla tamamladığını test eder."""
    supervisor = SupervisorAjan(max_revizyon=3)
    sonuc = supervisor.gorevi_orkestre_et("Dizi içindeki en büyük toplamlı ardışık alt diziyi bulma")

    assert sonuc["tamamlandi"] is True
    assert sonuc["toplam_revizyon"] >= 2
    assert sonuc["nihai_kalite_skoru"] >= 95.0
    assert len(sonuc["adim_gecmisi"]) >= 4


def test_supervisor_ajan_nihai_kod_calisabilirlik():
    """Üretilen nihai Python kodunun derlenip negatif sayılarda doğru sonuç ürettiğini test eder."""
    supervisor = SupervisorAjan()
    sonuc = supervisor.gorevi_orkestre_et("Max Subarray")
    kod = sonuc["nihai_kod"]

    yerel_alan = {}
    exec(kod, yerel_alan)
    fn = yerel_alan["max_alt_dizi"]

    # Test 1: Normal dizi
    assert fn([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    # Test 2: Tamamı negatif kritik sınır durum
    assert fn([-8, -3, -6, -2, -5, -4]) == -2


def test_benchmark_karsilastir_metrikleri():
    """SupervisorAjan kıyaslama metriklerinin eksiksiz olduğunu test eder."""
    supervisor = SupervisorAjan()
    bench = supervisor.benchmark_karsilastir()

    assert len(bench["metrikler"]) == 4
    assert bench["supervisor_worker_ajanlar"][0] > bench["tekil_genel_ajan"][0]


def test_multi_agent_gorsellestirici_pano():
    """MultiAgentGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    supervisor = SupervisorAjan()
    rapor = supervisor.gorevi_orkestre_et("Max Subarray")
    bench = supervisor.benchmark_karsilastir()

    gorsellestirici = MultiAgentGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_multi_agent_pano.png")
        gorsellestirici.pano_olustur(rapor, bench, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
