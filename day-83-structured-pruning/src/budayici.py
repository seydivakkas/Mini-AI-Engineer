"""
L1/L2 Norm Tabanlı Yapısal Filtre ve Kanal Budayıcı
--------------------------------------------------
Evrişimli filtrelerin önem derecelerini L1/L2 normları ile hesaplayan,
en düşük normlu filtreleri budayan ve sonraki katman dikişlerini (Layer Stitching)
fiziksel olarak yeniden inşa eden budama motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn

from .model import BudanabilirVisionCNN


class YapisalFiltreBudayici:
    """
    L1 ve L2 normu ile yapısal filtre budama ve fiziksel katman küçültme motoru.
    """
    @staticmethod
    def filtre_norm_hesapla(conv_katmani: nn.Conv2d, norm_tipi: str = "L1") -> torch.Tensor:
        """
        Her bir filtrenin (output channel) L1 veya L2 normunu hesaplar.
        Girdi: [C_out, C_in, K, K]
        Çıktı: [C_out] önem skorları
        """
        w = conv_katmani.weight.data
        if norm_tipi.upper() == "L1":
            # |W| toplamı
            skorlar = torch.sum(torch.abs(w), dim=(1, 2, 3))
        elif norm_tipi.upper() == "L2":
            # sqrt(sum(W^2))
            skorlar = torch.sqrt(torch.sum(w ** 2, dim=(1, 2, 3)))
        else:
            raise ValueError(f"Bilinmeyen norm tipi: {norm_tipi}. 'L1' veya 'L2' kullanın.")
        return skorlar

    @staticmethod
    def korunacak_indeksleri_sec(skorlar: torch.Tensor, budama_orani: float) -> torch.Tensor:
        """
        En yüksek skora sahip filtrelerin indekslerini seçer.
        """
        assert 0.0 <= budama_orani < 1.0, "Budama oranı [0, 1) aralığında olmalıdır!"
        c_out = len(skorlar)
        korunacak_sayi = max(1, int(c_out * (1.0 - budama_orani)))

        _, siralı_indeksler = torch.topk(skorlar, k=korunacak_sayi, largest=True)
        # Orijinal sıralamayı korumak için sıralayalım
        korunan_indeksler, _ = torch.sort(siralı_indeksler)
        return korunan_indeksler

    @classmethod
    def modeli_yapisal_buda(
        cls,
        eski_model: BudanabilirVisionCNN,
        budama_orani: float,
        norm_tipi: str = "L1"
    ) -> Tuple[BudanabilirVisionCNN, Dict[str, Any]]:
        """
        Orijinal modeli yapısal olarak budar ve fiziksel olarak daha küçük yeni bir model üretir.
        """
        cihaz = next(eski_model.parameters()).device

        # 1. Her katman için L1/L2 skorlarını ve tutulacak indeksleri hesapla
        skorlar_1 = cls.filtre_norm_hesapla(eski_model.conv1, norm_tipi)
        skorlar_2 = cls.filtre_norm_hesapla(eski_model.conv2, norm_tipi)
        skorlar_3 = cls.filtre_norm_hesapla(eski_model.conv3, norm_tipi)

        k1 = cls.korunacak_indeksleri_sec(skorlar_1, budama_orani)
        k2 = cls.korunacak_indeksleri_sec(skorlar_2, budama_orani)
        k3 = cls.korunacak_indeksleri_sec(skorlar_3, budama_orani)

        yeni_kanallar = [len(k1), len(k2), len(k3)]

        # 2. Yeni fiziksel modeli oluştur
        yeni_model = BudanabilirVisionCNN(
            giris_kanali=eski_model.giris_kanali,
            sinif_sayisi=eski_model.sinif_sayisi,
            kanallar=yeni_kanallar
        ).to(cihaz)

        # 3. Ağırlıkları ve Katman Dikişlerini (Layer Stitching) Kopyala
        with torch.no_grad():
            # Aşama 1: Conv1 & BN1
            yeni_model.conv1.weight.data.copy_(eski_model.conv1.weight.data[k1, :, :, :])
            yeni_model.bn1.weight.data.copy_(eski_model.bn1.weight.data[k1])
            yeni_model.bn1.bias.data.copy_(eski_model.bn1.bias.data[k1])
            yeni_model.bn1.running_mean.data.copy_(eski_model.bn1.running_mean.data[k1])
            yeni_model.bn1.running_var.data.copy_(eski_model.bn1.running_var.data[k1])

            # Aşama 2: Conv2 (hem çıkış k2, hem giriş k1 filtrelenir) & BN2
            conv2_w = eski_model.conv2.weight.data[k2, :, :, :][:, k1, :, :]
            yeni_model.conv2.weight.data.copy_(conv2_w)
            yeni_model.bn2.weight.data.copy_(eski_model.bn2.weight.data[k2])
            yeni_model.bn2.bias.data.copy_(eski_model.bn2.bias.data[k2])
            yeni_model.bn2.running_mean.data.copy_(eski_model.bn2.running_mean.data[k2])
            yeni_model.bn2.running_var.data.copy_(eski_model.bn2.running_var.data[k2])

            # Aşama 3: Conv3 (hem çıkış k3, hem giriş k2 filtrelenir) & BN3
            conv3_w = eski_model.conv3.weight.data[k3, :, :, :][:, k2, :, :]
            yeni_model.conv3.weight.data.copy_(conv3_w)
            yeni_model.bn3.weight.data.copy_(eski_model.bn3.weight.data[k3])
            yeni_model.bn3.bias.data.copy_(eski_model.bn3.bias.data[k3])
            yeni_model.bn3.running_mean.data.copy_(eski_model.bn3.running_mean.data[k3])
            yeni_model.bn3.running_var.data.copy_(eski_model.bn3.running_var.data[k3])

            # Sınıflandırıcı Kafa: FC (giriş boyutu k3 filtrelenir)
            yeni_model.fc.weight.data.copy_(eski_model.fc.weight.data[:, k3])
            yeni_model.fc.bias.data.copy_(eski_model.fc.bias.data)

        rapor = {
            "budama_orani": budama_orani,
            "norm_tipi": norm_tipi,
            "eski_kanallar": eski_model.kanallar,
            "yeni_kanallar": yeni_kanallar,
            "katman_skorlari": {
                "conv1": skorlar_1.cpu().numpy(),
                "conv2": skorlar_2.cpu().numpy(),
                "conv3": skorlar_3.cpu().numpy()
            },
            "korunan_indeksler": {
                "conv1": k1.cpu().numpy(),
                "conv2": k2.cpu().numpy(),
                "conv3": k3.cpu().numpy()
            }
        }

        return yeni_model, rapor
