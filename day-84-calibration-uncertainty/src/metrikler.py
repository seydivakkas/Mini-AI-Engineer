"""
Olasılık Kalibrasyonu ve Belirsizlik Metrikleri
----------------------------------------------
Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
Negative Log-Likelihood (NLL), Brier Score ve Güvenilirlik Diyagramı (Reliability Diagram) motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np


class KalibrasyonMetrikleri:
    """
    Modelin tahmin güveni ile gerçek doğruluğu arasındaki uyuşmazlığı ölçen metrik sınıfı.
    """
    @staticmethod
    def hesapla_tum_metrikler(
        logitler: torch.Tensor,
        etiketler: torch.Tensor,
        n_bins: int = 15
    ) -> Dict[str, Any]:
        """
        Logitler ve gerçek etiketler üzerinden ECE, MCE, NLL, Brier Score ve Bin istatistiklerini hesaplar.
        """
        with torch.no_grad():
            olasiliklar = F.softmax(logitler, dim=-1)
            guvenler, tahminler = torch.max(olasiliklar, dim=-1)
            dogrular = (tahminler == etiketler).float()

            n_ornek = logitler.size(0)

            # NLL (Negative Log-Likelihood)
            nll = F.cross_entropy(logitler, etiketler).item()

            # Brier Score: sum((p - y_onehot)^2) / N
            bir_sicak = F.one_hot(etiketler, num_classes=logitler.size(-1)).float()
            brier = torch.mean(torch.sum((olasiliklar - bir_sicak) ** 2, dim=-1)).item()

            # Binning İşlemi
            bin_sinirlari = torch.linspace(0, 1, n_bins + 1)
            bin_alt = bin_sinirlari[:-1]
            bin_ust = bin_sinirlari[1:]

            ece = 0.0
            mce = 0.0

            bin_dogruluklari = []
            bin_guvenleri = []
            bin_ornek_sayilari = []
            bin_farklari = []

            for alt, ust in zip(bin_alt, bin_ust):
                # Örneklerin bu aralığa düşme maskesi
                maske = (guvenler > alt) & (guvenler <= ust)
                bin_ornek = maske.sum().item()
                bin_ornek_sayilari.append(bin_ornek)

                if bin_ornek > 0:
                    bin_acc = dogrular[maske].mean().item()
                    bin_conf = guvenler[maske].mean().item()
                    fark = abs(bin_acc - bin_conf)

                    ece += (bin_ornek / n_ornek) * fark
                    mce = max(mce, fark)

                    bin_dogruluklari.append(bin_acc)
                    bin_guvenleri.append(bin_conf)
                    bin_farklari.append(fark)
                else:
                    bin_dogruluklari.append(0.0)
                    bin_guvenleri.append((alt.item() + ust.item()) / 2.0)
                    bin_farklari.append(0.0)

            dogruluk = dogrular.mean().item() * 100.0

        return {
            "ece": ece * 100.0,              # Yüzde cinsinden (%)
            "mce": mce * 100.0,              # Yüzde cinsinden (%)
            "nll": nll,
            "brier_score": brier,
            "dogruluk": dogruluk,
            "n_bins": n_bins,
            "bin_sinirlari": bin_sinirlari.numpy(),
            "bin_dogruluklari": np.array(bin_dogruluklari),
            "bin_guvenleri": np.array(bin_guvenleri),
            "bin_ornek_sayilari": np.array(bin_ornek_sayilari),
            "bin_farklari": np.array(bin_farklari),
            "tum_guvenler": guvenler.cpu().numpy(),
            "tum_dogrular": dogrular.cpu().numpy()
        }
