"""
Optuna Otomatik Hiperparametre Optimizasyonu Birim Testleri
------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import optuna

from src.tpe_motoru import MatematikselTPESampler, MedyanBudayici
from src.model import ParametrikVisionModeli
from src.optuna_optimize import OptunaHPOVurucu


def test_matematiksel_tpe_ornekleme_sinirlari():
    sampler = MatematikselTPESampler(gama=0.25, aday_sayisi=32, tohum=42)
    gecmis_x = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    gecmis_y = [0.5, 0.4, 0.8, 1.2, 1.5, 2.0]

    val_lin = sampler.ornekle(gecmis_x, gecmis_y, alt_sinir=0.0001, ust_sinir=0.1, log_olcek=False)
    assert 0.0001 <= val_lin <= 0.1

    val_log = sampler.ornekle(gecmis_x, gecmis_y, alt_sinir=0.0001, ust_sinir=0.1, log_olcek=True)
    assert 0.0001 <= val_log <= 0.1


def test_matematiksel_tpe_oran_maksimizasyonu():
    sampler = MatematikselTPESampler(gama=0.25, aday_sayisi=64, tohum=42)
    # En iyi sonuçlar (düşük y) 0.001 civarında
    gecmis_x = [0.001, 0.0012, 0.0009, 0.05, 0.08, 0.1]
    gecmis_y = [0.10, 0.12, 0.11, 2.5, 3.1, 4.0]

    val = sampler.ornekle(gecmis_x, gecmis_y, alt_sinir=0.0001, ust_sinir=0.2, log_olcek=True)
    # Örneklenen değer kötü bölgeden (0.1) ziyade iyi bölgeye (0.001) daha yakın olmalıdır
    assert val < 0.05


def test_medyan_budayici_karar():
    budayici = MedyanBudayici(baslangic_adimi=2)
    budayici.adim_raporla(step=2, deger=1.0)
    budayici.adim_raporla(step=2, deger=2.0)
    budayici.adim_raporla(step=2, deger=3.0)  # Medyan = 2.0

    # Minimizasyon modunda (Loss): 2.5 > 2.0 -> Budanmalı
    assert budayici.budanmali_mi(step=2, guncel_deger=2.5, mod="min") is True
    # 1.5 < 2.0 -> Budanmamalı
    assert budayici.budanmali_mi(step=2, guncel_deger=1.5, mod="min") is False


def test_parametrik_model_ileri_gecis():
    model = ParametrikVisionModeli(giris_kanali=3, sinif_sayisi=10, taban_kanal=16, dropout_orani=0.2)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
    assert not torch.isnan(out).any()


def test_optuna_study_olusturma():
    hpo = OptunaHPOVurucu(calisma_adi="Test_Study", hedef_yon="minimize", tohum=123)
    assert hpo.study.study_name == "Test_Study"
    assert hpo.study.direction == optuna.study.StudyDirection.MINIMIZE


def test_optuna_hpo_kisa_optimizasyon():
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    tr_loader = DataLoader(TensorDataset(x, y), batch_size=10)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=10)

    hpo = OptunaHPOVurucu(calisma_adi="Kisa_Test", hedef_yon="minimize", tohum=42, startup_deneme_sayisi=2)
    study = hpo.optimize_et(tr_loader, val_loader, deneme_sayisi=3, epok_sayisi=2, cihaz="cpu")

    assert len(study.trials) == 3


def test_optuna_calisma_ozeti_cikti_yapisi():
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    tr_loader = DataLoader(TensorDataset(x, y), batch_size=10)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=10)

    hpo = OptunaHPOVurucu(calisma_adi="Ozet_Test", hedef_yon="minimize", tohum=42, startup_deneme_sayisi=2)
    _ = hpo.optimize_et(tr_loader, val_loader, deneme_sayisi=3, epok_sayisi=2, cihaz="cpu")

    ozet = hpo.calisma_ozeti()
    assert "toplam_deneme" in ozet and ozet["toplam_deneme"] == 3
    assert "tamamlanan_sayisi" in ozet
    assert "en_iyi_deger" in ozet
    assert "en_iyi_parametreler" in ozet


def test_optuna_pruning_mekanizmasi():
    def budayan_objective(trial: optuna.Trial):
        trial.report(100.0, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return 100.0

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.ThresholdPruner(upper=50.0)
    )
    study.optimize(budayan_objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
