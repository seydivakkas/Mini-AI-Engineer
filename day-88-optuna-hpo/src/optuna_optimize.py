"""
Endüstriyel Optuna HPO ve Çalışma (Study) Yöneticisi
---------------------------------------------------
TPE Örnekleyici (TPESampler) ve Medyan Budayıcı (MedianPruner) ile otomatik
hiperparametre optimizasyonu, arama uzayı tanımları ve çalışma analitiği.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
from torch.utils.data import DataLoader

from .model import ParametrikVisionModeli


class OptunaHPOVurucu:
    """
    Optuna ile otomatik hiperparametre optimizasyonu yürüten çalışma yöneticisi.
    """
    def __init__(
        self,
        calisma_adi: str = "Vision_TPE_Optimizasyonu",
        hedef_yon: str = "minimize",
        tohum: int = 42,
        startup_deneme_sayisi: int = 4,
        warmup_epok_sayisi: int = 2
    ):
        self.calisma_adi = calisma_adi
        self.hedef_yon = hedef_yon
        self.tohum = tohum

        # TPE Sampler ve Median Pruner
        self.sampler = TPESampler(seed=tohum)
        self.pruner = MedianPruner(
            n_startup_trials=startup_deneme_sayisi,
            n_warmup_steps=warmup_epok_sayisi,
            interval_steps=1
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            study_name=calisma_adi,
            direction=hedef_yon,
            sampler=self.sampler,
            pruner=self.pruner
        )

    def optimize_et(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        deneme_sayisi: int = 15,
        epok_sayisi: int = 8,
        cihaz: str = "cpu"
    ) -> optuna.Study:
        """
        Arama uzayını tanımlar ve belirtilen deneme sayısı kadar optimizasyon koşturur.
        """
        def objective(trial: optuna.Trial) -> float:
            # Hiperparametre Arama Uzayı
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            opt_tipi = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            taban_kanal = trial.suggest_categorical("taban_kanal", [16, 32, 48])
            dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)

            model = ParametrikVisionModeli(
                sinif_sayisi=10,
                taban_kanal=taban_kanal,
                dropout_orani=dropout
            )

            val_loss = ParametrikVisionModeli.egit_ve_degerlendir(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                optimizator_tipi=opt_tipi,
                weight_decay=wd,
                epok_sayisi=epok_sayisi,
                trial=trial,
                cihaz=cihaz
            )
            return val_loss

        self.study.optimize(objective, n_trials=deneme_sayisi)
        return self.study

    def calisma_ozeti(self) -> Dict[str, Any]:
        """
        Çalışmanın tamamlanan, budanan denemelerini ve en iyi hiperparametreleri döner.
        """
        tamamlanan = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        budanan = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]

        df_trials = self.study.trials_dataframe()

        # Hiperparametre önem dereceleri
        try:
            onemler = optuna.importance.get_param_importances(self.study)
        except Exception:
            onemler = {}

        return {
            "toplam_deneme": len(self.study.trials),
            "tamamlanan_sayisi": len(tamamlanan),
            "budanan_sayisi": len(budanan),
            "en_iyi_deger": self.study.best_value,
            "en_iyi_parametreler": self.study.best_params,
            "df_trials": df_trials,
            "param_importances": onemler
        }
