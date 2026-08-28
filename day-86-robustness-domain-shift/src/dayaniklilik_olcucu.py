"""
Model Dayanıklılık ve Bozulma Ölçümleyicisi (mCE & Rel-mCE)
----------------------------------------------------------
Hendrycks & Dietterich (ICLR 2019) metrik standartları:
Mean Corruption Error (mCE), Relative mCE ve Şiddet Seviyesi (Severity 1..5) analizi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .bozulma_motoru import GorselBozulmaMotoru


class DayaniklilikOlcucu:
    """
    Modelleri temiz ve 8 farklı bozulma altında 5 şiddet seviyesinde stres testine tabi tutan motor.
    """
    @staticmethod
    def temiz_dogruluk_olc(model: nn.Module, loader: DataLoader, cihaz: str = "cpu") -> float:
        model = model.to(cihaz).eval()
        toplam_dogru = 0
        toplam_ornek = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(cihaz), y.to(cihaz)
                tahminler = model(x).argmax(dim=-1)
                toplam_dogru += (tahminler == y).sum().item()
                toplam_ornek += x.size(0)

        return (toplam_dogru / max(1, toplam_ornek)) * 100.0

    @classmethod
    def kapsamli_stres_testi(
        cls,
        model: nn.Module,
        loader: DataLoader,
        cihaz: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Modeli tüm bozulma tiplerinde (C=8) ve tüm şiddet seviyelerinde (s=1..5) değerlendirir.
        """
        model = model.to(cihaz).eval()
        temiz_acc = cls.temiz_dogruluk_olc(model, loader, cihaz)
        temiz_hata = 100.0 - temiz_acc

        bozulmalar = GorselBozulmaMotoru.tum_bozulma_fonksiyonlari()
        
        # Sonuç tabloları: {bozulma_adi: [acc_s1, acc_s2, acc_s3, acc_s4, acc_s5]}
        bozulma_dogruluklari: Dict[str, List[float]] = {}
        bozulma_hatalari: Dict[str, List[float]] = {}

        for b_adi, b_fn in bozulmalar.items():
            acc_listesi = []
            err_listesi = []
            for s in range(1, 6):
                dogru_sayisi = 0
                toplam_ornek = 0
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(cihaz), y.to(cihaz)
                        # Bozulmayı uygula
                        x_bozuk = b_fn(x, siddet=s)
                        tahmin = model(x_bozuk).argmax(dim=-1)
                        dogru_sayisi += (tahmin == y).sum().item()
                        toplam_ornek += x.size(0)

                acc = (dogru_sayisi / max(1, toplam_ornek)) * 100.0
                err = 100.0 - acc
                acc_listesi.append(acc)
                err_listesi.append(err)

            bozulma_dogruluklari[b_adi] = acc_listesi
            bozulma_hatalari[b_adi] = err_listesi

        # Ortalama Bozulma Hatası (mCE): Tüm bozulmalar ve tüm şiddetlerin ortalama hatası
        tum_hatalar = [np.mean(errs) for errs in bozulma_hatalari.values()]
        mce = float(np.mean(tum_hatalar))
        macc = 100.0 - mce
        rel_mce = mce - temiz_hata

        # Şiddet bazında ortalama doğruluk eğrisi (s=1..5)
        siddet_egrisi = []
        for s_idx in range(5):
            s_acc = np.mean([accs[s_idx] for accs in bozulma_dogruluklari.values()])
            siddet_egrisi.append(float(s_acc))

        return {
            "temiz_dogruluk": temiz_acc,
            "temiz_hata": temiz_hata,
            "mce": mce,
            "macc": macc,
            "rel_mce": rel_mce,
            "bozulma_dogruluklari": bozulma_dogruluklari,
            "bozulma_hatalari": bozulma_hatalari,
            "siddet_egrisi": np.array(siddet_egrisi)
        }
