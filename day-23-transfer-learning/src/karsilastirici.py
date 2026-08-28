"""Transfer Öğrenme Kıyaslama ve Karşılaştırma Deney Motoru.

Bu modül; Sıfırdan Eğitim (Scratch), Öznitelik Çıkarma (Feature Extraction)
ve İnce Ayar (Fine-Tuning) yaklaşımlarını aynı veri seti üzerinde eğiterek
yakınsama hızı, eğitilebilir parametre oranı ve test doğruluğunu karşılaştırır.
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader

from src.model_secici import TransferModelSecici
from src.egitici import TransferEgitici, TransferEgitimSonucu


class TransferKarsilastirici:
    """Farklı transfer öğrenme stratejilerini eğiten ve karşılaştıran sınıf."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def karsilastirmali_deney_kos(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int = 4,
        epochs: int = 15,
    ) -> List[TransferEgitimSonucu]:
        """Üç temel transfer stratejisini sırayla koşturur ve sonuç listesini döner."""
        deneyler = [
            ("ResNet18 (Scratch / Sıfırdan)", "scratch", False),
            ("ResNet18 (Feature Extraction)", "feature_extraction", False),
            ("ResNet18 (Fine-Tuning / İnce Ayar)", "fine_tuning", False),
            ("EfficientNet-B0 (Feature Extraction)", "feature_extraction_eff", False),
        ]

        sonuclar: List[TransferEgitimSonucu] = []

        for model_adi, strat, pretrained in deneyler:
            print(f"[*] Deney Başlatılıyor: {model_adi}...")

            if strat == "feature_extraction_eff":
                model = TransferModelSecici.efficientnet_b0_olustur(
                    num_classes=num_classes, pretrained=pretrained, strateji="feature_extraction"
                )
                param_gruplari = None
            elif strat == "fine_tuning":
                model = TransferModelSecici.resnet18_olustur(
                    num_classes=num_classes, pretrained=pretrained, strateji="fine_tuning"
                )
                param_gruplari = TransferModelSecici.ayrisik_parametre_gruplari(
                    model, lr_omurga=1e-4, lr_baslik=1e-3
                )
            elif strat == "feature_extraction":
                model = TransferModelSecici.resnet18_olustur(
                    num_classes=num_classes, pretrained=pretrained, strateji="feature_extraction"
                )
                param_gruplari = None
            else:  # scratch
                model = TransferModelSecici.resnet18_olustur(
                    num_classes=num_classes, pretrained=pretrained, strateji="scratch"
                )
                param_gruplari = None

            egitici = TransferEgitici(model, device=self.device)
            t0 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
            t1 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None

            import time
            start_time = time.perf_counter()

            tarihce, _ = egitici.egit(
                train_loader=train_loader,
                val_loader=val_loader,
                param_gruplari=param_gruplari,
                lr_varsayilan=0.001,
                epochs=epochs,
                patience=6,
            )
            gecen_sure = time.perf_counter() - start_time

            sonuc = egitici.degerlendir(
                test_loader=test_loader,
                tarihce=tarihce,
                model_adi=model_adi,
                strateji=strat,
                egitim_suresi_sn=gecen_sure,
            )
            sonuclar.append(sonuc)

        return sonuclar
