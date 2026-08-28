"""Transfer Öğrenme Model Seçici ve Omurga Düzenleyici Modülü.

Bu modül; ResNet ve EfficientNet gibi endüstri standardı önceden eğitilmiş
(Pretrained ImageNet) omurgaları yükler, katman dondurma (Freeze), katman kilidi açma (Unfreeze),
özel sınıflandırıcı başlığı ekleme ve ayrıştırılmış öğrenme oranları için parametre gruplandırmayı yönetir.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models


class TransferModelSecici:
    """Önceden eğitilmiş derin öğrenme modellerini transfer öğrenme için hazırlayan sınıf."""

    @staticmethod
    def dondur_omurga(model: nn.Module) -> None:
        """Modelin tüm parametrelerini dondurur (requires_grad = False)."""
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def resnet18_olustur(
        num_classes: int = 4,
        pretrained: bool = False,
        strateji: str = "feature_extraction",
        dropout_rate: float = 0.3,
    ) -> nn.Module:
        """ResNet18 modelini belirtilen transfer stratejisine göre yapılandırır.

        Args:
            num_classes: Hedef sınıf sayısı.
            pretrained: ImageNet ağırlıklarının yüklenip yüklenmeyeceği.
            strateji: 'feature_extraction', 'fine_tuning' veya 'scratch'.
            dropout_rate: Sınıflandırıcı başlığı dropout oranı.

        Returns:
            nn.Module: Yapılandırılmış PyTorch modeli.
        """
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        in_features = model.fc.in_features  # 512

        if strateji == "feature_extraction":
            # Tüm omurgayı dondur
            for param in model.parameters():
                param.requires_grad = False

            # Özel sınıflandırıcı başlığı (Custom Head) ekle (yalnızca bu eğitilecek)
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

        elif strateji == "fine_tuning":
            # Önce tüm omurgayı dondur, ardından son blokların (layer4) kilidini aç
            for param in model.parameters():
                param.requires_grad = False

            for param in model.layer4.parameters():
                param.requires_grad = True

            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

        else:  # scratch (sıfırdan rastgele ağırlıklarla eğitim)
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )
            for param in model.parameters():
                param.requires_grad = True

        return model

    @staticmethod
    def efficientnet_b0_olustur(
        num_classes: int = 4,
        pretrained: bool = False,
        strateji: str = "feature_extraction",
        dropout_rate: float = 0.3,
    ) -> nn.Module:
        """EfficientNet-B0 modelini belirtilen transfer stratejisine göre yapılandırır."""
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features  # 1280

        if strateji == "feature_extraction":
            for param in model.parameters():
                param.requires_grad = False

            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

        elif strateji == "fine_tuning":
            for param in model.parameters():
                param.requires_grad = False

            # Son evrişim bloklarının (features[7]) kilidini aç
            for param in model.features[7].parameters():
                param.requires_grad = True

            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

        else:  # scratch
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )
            for param in model.parameters():
                param.requires_grad = True

        return model

    @staticmethod
    def ayrisik_parametre_gruplari(
        model: nn.Module,
        lr_omurga: float = 1e-4,
        lr_baslik: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Union[List[nn.Parameter], float]]]:
        """İnce ayar için omurga ve başlık parametrelerini ayrıştırılmış LR ile gruplar."""
        baslik_param_ids = set()
        baslik_params = []

        # Başlık katmanını tespit et (ResNet için fc, EfficientNet için classifier)
        hedef_baslik = getattr(model, "fc", None) or getattr(model, "classifier", None)
        if hedef_baslik is not None:
            for p in hedef_baslik.parameters():
                if p.requires_grad:
                    baslik_params.append(p)
                    baslik_param_ids.add(id(p))

        omurga_params = [
            p for p in model.parameters() if p.requires_grad and id(p) not in baslik_param_ids
        ]

        gruplar = []
        if omurga_params:
            gruplar.append({
                "params": omurga_params,
                "lr": lr_omurga,
                "weight_decay": weight_decay,
            })
        if baslik_params:
            gruplar.append({
                "params": baslik_params,
                "lr": lr_baslik,
                "weight_decay": weight_decay,
            })

        return gruplar

    @staticmethod
    def parametre_ozeti(model: nn.Module) -> Dict[str, int]:
        """Modelin toplam, eğitilebilir ve dondurulmuş parametre sayılarını döndürür."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
