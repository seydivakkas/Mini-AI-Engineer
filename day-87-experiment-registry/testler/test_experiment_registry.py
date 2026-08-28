"""
Merkezi Deney Takibi ve Artefakt Kayıt Sistemi Birim Testleri
-------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import shutil
import pytest
import torch
import pandas as pd

from src.takip_motoru import MerkeziDeneyTakipMotoru
from src.karsilastirici import DeneyKarsilastirici
from src.model import DeneyVisionModeli


@pytest.fixture
def gecici_motor(tmp_path):
    depo_yolu = str(tmp_path / ".test_deney_deposu")
    motor = MerkeziDeneyTakipMotoru(depo_dizini=depo_yolu)
    yield motor
    if os.path.exists(depo_yolu):
        shutil.rmtree(depo_yolu, ignore_errors=True)


def test_deney_olusturma_ve_kimlik(gecici_motor):
    exp_id1 = gecici_motor.deney_olustur_veya_getir("Test_Deneyi_1")
    exp_id2 = gecici_motor.deney_olustur_veya_getir("Test_Deneyi_1")
    exp_id3 = gecici_motor.deney_olustur_veya_getir("Test_Deneyi_2")

    assert exp_id1 == exp_id2
    assert exp_id1 != exp_id3


def test_kosu_parametre_loglama(gecici_motor):
    kosu = gecici_motor.start_run(deney_adi="Param_Testi", kosu_adi="Kosu_1")
    kosu.log_param("lr", 0.001)
    kosu.log_params({"optimizer": "adamw", "batch_size": 32})
    gecici_motor.end_run()

    kosular = gecici_motor.tum_kosulari_getir("Param_Testi")
    assert len(kosular) == 1
    assert kosular[0]["params"]["lr"] == "0.001"
    assert kosular[0]["params"]["optimizer"] == "adamw"
    assert kosular[0]["params"]["batch_size"] == "32"


def test_kosu_metrik_zaman_serisi_loglama(gecici_motor):
    kosu = gecici_motor.start_run(deney_adi="Metrik_Testi")
    kosu.log_metric("loss", 0.95, step=1)
    kosu.log_metric("loss", 0.55, step=2)
    kosu.log_metric("loss", 0.25, step=3)
    gecici_motor.end_run()

    kosular = gecici_motor.tum_kosulari_getir("Metrik_Testi")
    assert len(kosular) == 1
    assert kosular[0]["metrics"]["loss"] == 0.25
    gecmis = kosular[0]["metric_history"]["loss"]
    assert len(gecmis) == 3
    assert gecmis[0]["value"] == 0.95
    assert gecmis[2]["value"] == 0.25


def test_kosu_artefakt_kayit(gecici_motor, tmp_path):
    dosya = tmp_path / "rapor.txt"
    dosya.write_text("Test Raporu")

    kosu = gecici_motor.start_run(deney_adi="Artefakt_Testi")
    kayitli_yol = kosu.log_artifact(str(dosya))
    gecici_motor.end_run()

    assert os.path.exists(kayitli_yol)
    assert open(kayitli_yol, "r").read() == "Test Raporu"


def test_kosu_model_kayit_ve_yukleme(gecici_motor):
    model = DeneyVisionModeli(giris_kanali=3, sinif_sayisi=5, taban_kanal=8)
    kosu = gecici_motor.start_run(deney_adi="Model_Testi")
    model_yolu = kosu.log_model(model, "model_v1.pt")
    gecici_motor.end_run()

    assert os.path.exists(model_yolu)
    yuklenen_state = torch.load(model_yolu, weights_only=True)
    yeni_model = DeneyVisionModeli(giris_kanali=3, sinif_sayisi=5, taban_kanal=8)
    yeni_model.load_state_dict(yuklenen_state)
    assert sum(p.numel() for p in yeni_model.parameters()) == sum(p.numel() for p in model.parameters())


def test_tum_kosulari_getir_sorgusu(gecici_motor):
    k1 = gecici_motor.start_run(deney_adi="Coklu_Kosu", kosu_adi="K1")
    k1.log_metric("acc", 85.0, step=1)
    gecici_motor.end_run()

    k2 = gecici_motor.start_run(deney_adi="Coklu_Kosu", kosu_adi="K2")
    k2.log_metric("acc", 92.5, step=1)
    gecici_motor.end_run()

    kosular = gecici_motor.tum_kosulari_getir("Coklu_Kosu")
    assert len(kosular) == 2


def test_karsilastirma_tablosu_ve_siralama():
    kosular = [
        {
            "run_id": "r1",
            "durum": "FINISHED",
            "baslangic_zamani": 100.0,
            "bitis_zamani": 110.0,
            "params": {"lr": "0.01"},
            "tags": {"run_name": "Kosu_A"},
            "metrics": {"val_acc": 80.0},
            "metric_history": {}
        },
        {
            "run_id": "r2",
            "durum": "FINISHED",
            "baslangic_zamani": 120.0,
            "bitis_zamani": 135.0,
            "params": {"lr": "0.001"},
            "tags": {"run_name": "Kosu_B"},
            "metrics": {"val_acc": 95.0},
            "metric_history": {}
        }
    ]

    df = DeneyKarsilastirici.karsilastirma_tablosu(kosular)
    assert len(df) == 2
    assert df.iloc[0]["run_name"] == "Kosu_B"
    assert df.iloc[0]["m_val_acc"] == 95.0


def test_pareto_optimal_hesaplama():
    df = pd.DataFrame([
        {"run_name": "Model_Kucuk", "p_param_count": 1000, "m_val_acc": 80.0},
        {"run_name": "Model_Orta", "p_param_count": 5000, "m_val_acc": 90.0},
        {"run_name": "Model_Kotü", "p_param_count": 8000, "m_val_acc": 85.0},  # Dominated
        {"run_name": "Model_Buyuk", "p_param_count": 10000, "m_val_acc": 95.0},
    ])

    pareto_df = DeneyKarsilastirici.pareto_optimal_noktalari(df, x_kolon="p_param_count", y_kolon="m_val_acc")
    assert len(pareto_df) == 3
    assert "Model_Kotü" not in pareto_df["run_name"].values
