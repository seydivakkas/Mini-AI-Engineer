"""
Determinizm ve Tekrarlanabilirlik Dogrulayicisi (Reproducibility Validator)
=========================================================================
Ayni konfigürasyon ve tohumla calistirilan iki bagimsiz egitim kosusunun (Run A vs Run B)
agirlik, kayip ve dogruluk degerlerini bit-for-bit karsilastirir.
"""

from typing import Dict, Any, List
import numpy as np
from src.konfigurasyon_semasi import KokKonfigurasyon
from src.egitim_motoru import TekrarlanabilirEgitici


class DeterminizmDogrulayici:
    """
    Iki bagimsiz egitim surecinin birebir ayni sonuclari verdigini kanitlayan dogrulama motoru.
    """

    @classmethod
    def determinizm_testi_kos(
        cls,
        ana_config: KokKonfigurasyon,
        farkli_tohum_baseline: int = 99
    ) -> Dict[str, Any]:
        """
        Run A (Seed=S), Run B (Seed=S) ve Run C (Seed=S2) kosularini gerceklestirir ve karsilastirir.
        """
        # 1. Koşu A (Deterministik Hedef)
        egitici_a = TekrarlanabilirEgitici(ana_config)
        sonuc_a = egitici_a.egit()

        # 2. Koşu B (Deterministik Tekrar - Birebir Aynısı Olmalı)
        egitici_b = TekrarlanabilirEgitici(ana_config)
        sonuc_b = egitici_b.egit()

        # 3. Koşu C (Farklı Tohum - Baseline Varyansını Göstermek İçin)
        config_c = ana_config.model_copy(deep=True)
        config_c.egitim.tohum = farkli_tohum_baseline
        egitici_c = TekrarlanabilirEgitici(config_c)
        sonuc_c = egitici_c.egit()

        # Fark ve Eşdeğerlik Analizi
        kayip_farki_ab = [
            abs(la - lb) for la, lb in zip(sonuc_a["gecmis"]["train_loss"], sonuc_b["gecmis"]["train_loss"])
        ]
        val_kayip_farki_ab = [
            abs(la - lb) for la, lb in zip(sonuc_a["gecmis"]["val_loss"], sonuc_b["gecmis"]["val_loss"])
        ]
        acc_farki_ab = [
            abs(aa - ab) for aa, ab in zip(sonuc_a["gecmis"]["val_accuracy"], sonuc_b["gecmis"]["val_accuracy"])
        ]

        maks_kayip_farki_ab = float(max(kayip_farki_ab))
        maks_val_kayip_farki_ab = float(max(val_kayip_farki_ab))
        maks_acc_farki_ab = float(max(acc_farki_ab))

        agirlik_hash_eslesmesi = (sonuc_a["son_agirlik_hashi"] == sonuc_b["son_agirlik_hashi"])
        tum_epoch_hashleri_esit = (egitici_a.epoch_agirlik_hashleri == egitici_b.epoch_agirlik_hashleri)

        # Baseline ile fark
        kayip_farki_ac = [
            abs(la - lc) for la, lc in zip(sonuc_a["gecmis"]["train_loss"], sonuc_c["gecmis"]["train_loss"])
        ]
        maks_kayip_farki_ac = float(max(kayip_farki_ac))

        deterministik_basarili = (
            maks_kayip_farki_ab == 0.0 and
            maks_acc_farki_ab == 0.0 and
            agirlik_hash_eslesmesi and
            tum_epoch_hashleri_esit
        )

        return {
            "deterministik_basarili": deterministik_basarili,
            "maks_train_loss_delta_ab": round(maks_kayip_farki_ab, 9),
            "maks_val_loss_delta_ab": round(maks_val_kayip_farki_ab, 9),
            "maks_val_acc_delta_ab": round(maks_acc_farki_ab, 6),
            "agirlik_hash_eslesmesi": agirlik_hash_eslesmesi,
            "run_a_hash": sonuc_a["son_agirlik_hashi"],
            "run_b_hash": sonuc_b["son_agirlik_hashi"],
            "run_c_hash": sonuc_c["son_agirlik_hashi"],
            "maks_loss_farki_ac_baseline": round(maks_kayip_farki_ac, 6),
            "sonuc_a": sonuc_a,
            "sonuc_b": sonuc_b,
            "sonuc_c": sonuc_c
        }
