"""
Day 92: Veri Sözleşmesi Denetim Motoru (Data Contract Inspector)
---------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .sozlesme_kurallari import VeriSozlesmesi, KuralIhlali, IhlalSeviyesi


@dataclass
class DenetimSonucu:
    toplam_ornek: int
    gecerli_ornek_sayisi: int
    ihlal_listesi: List[KuralIhlali]
    istatistikler: Dict[str, float]
    sinif_dagilimi: Dict[int, int]
    nan_inf_sayisi: int
    min_deger: float
    maks_deger: float
    ortalama_deger: float
    standart_sapma: float

    @property
    def bloklayan_hata_var_mi(self) -> bool:
        return any(i.seviye == IhlalSeviyesi.BLOKE_EDICI for i in self.ihlal_listesi)

    @property
    def uyari_sayisi(self) -> int:
        return sum(1 for i in self.ihlal_listesi if i.seviye == IhlalSeviyesi.UYARI)


class VeriDenetleyici:
    """
    Eğitim öncesi tensör ve etiket yığınlarını katı VeriSozlesmesi kurallarına göre
    denetleyen, anomalileri, eksik/bozuk değerleri ve dağılım sorunlarını raporlayan motor.
    """

    def __init__(self, sozlesme: Optional[VeriSozlesmesi] = None):
        self.sozlesme = sozlesme or VeriSozlesmesi()

    def denetle(self, veriler: torch.Tensor, etiketler: Optional[torch.Tensor] = None) -> DenetimSonucu:
        """Veri setini sözleşme kurallarına göre uçtan uca tarar."""
        ihlal_listesi: List[KuralIhlali] = []
        toplam_ornek = len(veriler)

        # 1. Minimum Hacim Kontrolü
        if toplam_ornek < self.sozlesme.min_ornek_sayisi:
            ihlal_listesi.append(
                KuralIhlali(
                    kural_adi="MIN_ORNEK_SAYISI",
                    seviye=IhlalSeviyesi.BLOKE_EDICI,
                    mesaj=f"Veri seti hacmi ({toplam_ornek}) sözleşmedeki minimum sınırın ({self.sozlesme.min_ornek_sayisi}) altında!",
                    etkilenen_ornek_sayisi=toplam_ornek,
                    metrik_degeri=float(toplam_ornek),
                )
            )

        # 2. Şema ve Boyut Kontrolü (Shape)
        if veriler.ndim != 4:
            ihlal_listesi.append(
                KuralIhlali(
                    kural_adi="TENSOR_BOYUTU_HATASI",
                    seviye=IhlalSeviyesi.BLOKE_EDICI,
                    mesaj=f"Beklenen tensör boyutu 4D (N, C, H, W) ancak {veriler.ndim}D alındı.",
                    etkilenen_ornek_sayisi=toplam_ornek,
                )
            )
        else:
            _, c, h, w = veriler.shape
            beklenen_c, beklenen_h, beklenen_w = self.sozlesme.beklenen_sekil
            if (c, h, w) != (beklenen_c, beklenen_h, beklenen_w):
                ihlal_listesi.append(
                    KuralIhlali(
                        kural_adi="GIRDI_SEKIL_UYUMSUZLUGU",
                        seviye=IhlalSeviyesi.BLOKE_EDICI,
                        mesaj=f"Girdi çözünürlüğü ({c}, {h}, {w}) sözleşme ile uyumsuz. Beklenen: ({beklenen_c}, {beklenen_h}, {beklenen_w})",
                        etkilenen_ornek_sayisi=toplam_ornek,
                    )
                )

        # 3. Veri Tipi Kontrolü (Dtype)
        dtype_str = str(veriler.dtype).replace("torch.", "")
        if self.sozlesme.beklenen_dtype not in dtype_str:
            ihlal_listesi.append(
                KuralIhlali(
                    kural_adi="DTYPE_UYUMSUZLUGU",
                    seviye=IhlalSeviyesi.UYARI,
                    mesaj=f"Veri tipi '{dtype_str}' sözleşmedeki '{self.sozlesme.beklenen_dtype}' ile tam uyuşmuyor.",
                    etkilenen_ornek_sayisi=toplam_ornek,
                )
            )

        # 4. NaN / Inf Kontrolleri
        nan_sayisi = int(torch.isnan(veriler).sum().item())
        inf_sayisi = int(torch.isinf(veriler).sum().item())
        toplam_nan_inf = nan_sayisi + inf_sayisi

        if self.sozlesme.nan_inf_yasak and toplam_nan_inf > 0:
            ihlal_listesi.append(
                KuralIhlali(
                    kural_adi="NAN_INF_TESPITI",
                    seviye=IhlalSeviyesi.BLOKE_EDICI,
                    mesaj=f"Veri tensöründe {nan_sayisi} adet NaN ve {inf_sayisi} adet Inf değeri tespit edildi!",
                    etkilenen_ornek_sayisi=toplam_nan_inf,
                    metrik_degeri=float(toplam_nan_inf),
                )
            )

        # 5. Sayısal Sınırlar ve İstatistikler
        temiz_veriler = veriler[~torch.isnan(veriler) & ~torch.isinf(veriler)]
        if len(temiz_veriler) > 0:
            min_val = float(temiz_veriler.min().item())
            maks_val = float(temiz_veriler.max().item())
            ort_val = float(temiz_veriler.mean().item())
            std_val = float(temiz_veriler.std().item())
        else:
            min_val, maks_val, ort_val, std_val = 0.0, 0.0, 0.0, 0.0

        if min_val < self.sozlesme.min_deger_limiti or maks_val > self.sozlesme.maks_deger_limiti:
            ihlal_listesi.append(
                KuralIhlali(
                    kural_adi="DEGER_SINIRI_ASIMI",
                    seviye=IhlalSeviyesi.UYARI,
                    mesaj=f"Veri değer aralığı [{min_val:.2f}, {maks_val:.2f}] sözleşme limitleri [{self.sozlesme.min_deger_limiti}, {self.sozlesme.maks_deger_limiti}] dışına çıkıyor.",
                    metrik_degeri=float(maks_val - min_val),
                )
            )

        # 6. Sınıf Dağılımı ve Dengesizlik Kontrolleri
        sinif_dagilimi: Dict[int, int] = {}
        if etiketler is not None and len(etiketler) > 0:
            etiket_np = etiketler.cpu().numpy().astype(int)
            benzersiz, sayimlar = np.unique(etiket_np, return_counts=True)
            sinif_dagilimi = {int(k): int(v) for k, v in zip(benzersiz, sayimlar)}

            # Beklenen sınıf sayısı kontrolü
            if len(sinif_dagilimi) > self.sozlesme.beklenen_sinif_sayisi:
                ihlal_listesi.append(
                    KuralIhlali(
                        kural_adi="BEKLENMEYEN_SINIF_SAYISI",
                        seviye=IhlalSeviyesi.BLOKE_EDICI,
                        mesaj=f"Etiketlerde {len(sinif_dagilimi)} farklı sınıf bulundu, sözleşme maksimum {self.sozlesme.beklenen_sinif_sayisi} sınıf bekliyordu.",
                    )
                )

            # Sınıf Dengesizlik Oranı
            min_sayim = min(sayimlar)
            maks_sayim = max(sayimlar)
            dengesizlik_orani = float(maks_sayim / max(1, min_sayim))

            if dengesizlik_orani > self.sozlesme.maks_sinif_dengesizlik_orani:
                ihlal_listesi.append(
                    KuralIhlali(
                        kural_adi="ASIRI_SINIF_DENGESIZLIGI",
                        seviye=IhlalSeviyesi.UYARI,
                        mesaj=f"Sınıf dengesizlik oranı {dengesizlik_orani:.1f}x (Maksimum tolerans: {self.sozlesme.maks_sinif_dengesizlik_orani:.1f}x).",
                        metrik_degeri=dengesizlik_orani,
                    )
                )

            # Nadir Sınıf Kontrolü
            nadir_siniflar = [k for k, v in sinif_dagilimi.items() if v < self.sozlesme.nadir_sinif_min_ornek]
            if nadir_siniflar:
                ihlal_listesi.append(
                    KuralIhlali(
                        kural_adi="NADIR_SINIF_YETERSIZLIGI",
                        seviye=IhlalSeviyesi.UYARI,
                        mesaj=f"{len(nadir_siniflar)} adet sınıf {self.sozlesme.nadir_sinif_min_ornek} adetten az örneğe sahip (Sınıflar: {nadir_siniflar}).",
                    )
                )

        gecerli_ornek = toplam_ornek if toplam_nan_inf == 0 else max(0, toplam_ornek - toplam_nan_inf)

        return DenetimSonucu(
            toplam_ornek=toplam_ornek,
            gecerli_ornek_sayisi=gecerli_ornek,
            ihlal_listesi=ihlal_listesi,
            istatistikler={
                "min": min_val,
                "maks": maks_val,
                "ortalama": ort_val,
                "std": std_val,
            },
            sinif_dagilimi=sinif_dagilimi,
            nan_inf_sayisi=toplam_nan_inf,
            min_deger=min_val,
            maks_deger=maks_val,
            ortalama_deger=ort_val,
            standart_sapma=std_val,
        )
