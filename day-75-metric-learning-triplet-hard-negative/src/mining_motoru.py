"""
Online Triplet Madenciliği Motoru (Hard, Semi-Hard, Easy Mining)
----------------------------------------------------------------
FaceNet (Schroff et al. 2015) ve Hermans et al. (2017) stratejileriyle
batch içi üçlü madenciliği gerçekleştiren optimize modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn.functional as F


class TripletMadencisi:
    """
    Online Batch Triplet Madencisi.
    Desteklenen Stratejiler:
    - 'batch_all': Kayıp üreten (loss > 0) tüm üçlüleri toplar.
    - 'batch_hard': Her anchor için en uzak pozitifi ve en yakın negatifi seçer.
    - 'batch_semi_hard': d(a, p) < d(a, n) < d(a, p) + alpha aralığındaki yarı-zor negatifleri seçer.
    """
    def __init__(self, marjin: float = 0.3):
        self.marjin = marjin

    @staticmethod
    def ikili_mesafe_matrisi(gomulmeler: torch.Tensor, kareli: bool = True) -> torch.Tensor:
        """
        Gömülmeler arası (B, B) boyutunda Öklid mesafe matrisi hesaplar.
        L2 normalize vektörlerde: ||u - v||^2 = 2 - 2(u . v)
        """
        dot_product = torch.matmul(gomulmeler, gomulmeler.T)
        kareli_mesafe = 2.0 - 2.0 * dot_product
        kareli_mesafe = torch.clamp(kareli_mesafe, min=0.0)
        kareli_mesafe.fill_diagonal_(0.0)
        
        if kareli:
            return kareli_mesafe
        return torch.sqrt(kareli_mesafe)

    def madencilik_yap(
        self,
        gomulmeler: torch.Tensor,
        etiketler: torch.Tensor,
        strateji: str = "batch_semi_hard"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Batch içindeki gömülmelerden belirtilen stratejiye göre triplet kaybı hesaplar.
        """
        B = gomulmeler.size(0)
        cihaz = gomulmeler.device
        mesafe_matrisi = self.ikili_mesafe_matrisi(gomulmeler, kareli=False)

        # Pozitif ve Negatif Maskeleri
        y_kolon = etiketler.contiguous().view(-1, 1)
        ayni_sinif = torch.eq(y_kolon, y_kolon.T)
        farkli_sinif = torch.logical_not(ayni_sinif)
        kosegen_olmayan = torch.logical_not(torch.eye(B, dtype=torch.bool, device=cihaz))
        
        pozitif_maske = ayni_sinif & kosegen_olmayan
        negatif_maske = farkli_sinif

        # İstatistik Takibi
        d_ap_list = []
        d_an_list = []
        kolay_sayisi = 0
        yari_zor_sayisi = 0
        zor_sayisi = 0
        toplam_gecerli_triplet = 0

        if strateji == "batch_hard":
            # Her anchor için en zor pozitif (en uzak) ve en zor negatif (en yakın)
            # En uzak pozitif: max(d(a, p))
            mesafe_pos = mesafe_matrisi.clone()
            mesafe_pos[~pozitif_maske] = -1e9
            en_zor_p_mesafeleri, _ = torch.max(mesafe_pos, dim=1)

            # En yakın negatif: min(d(a, n))
            mesafe_neg = mesafe_matrisi.clone()
            mesafe_neg[~negatif_maske] = 1e9
            en_zor_n_mesafeleri, _ = torch.min(mesafe_neg, dim=1)

            # Geçerli anchor'ları filtrele (en az 1 pozitif ve 1 negatif içeren)
            gecerli_anchorlar = (pozitif_maske.sum(1) > 0) & (negatif_maske.sum(1) > 0)
            
            kayiplar = F.relu(en_zor_p_mesafeleri - en_zor_n_mesafeleri + self.marjin)
            aktif_kayiplar = kayiplar[gecerli_anchorlar]
            
            kayip = aktif_kayiplar.mean() if aktif_kayiplar.numel() > 0 else torch.tensor(0.0, device=cihaz, requires_grad=True)

            d_ap_mean = en_zor_p_mesafeleri[gecerli_anchorlar].mean().item() if gecerli_anchorlar.any() else 0.0
            d_an_mean = en_zor_n_mesafeleri[gecerli_anchorlar].mean().item() if gecerli_anchorlar.any() else 0.0

            istatistikler = {
                "toplam_triplet": int(gecerli_anchorlar.sum().item()),
                "aktif_triplet_orani": float((aktif_kayiplar > 0).float().mean().item()) if aktif_kayiplar.numel() > 0 else 0.0,
                "d_ap_ort": d_ap_mean,
                "d_an_ort": d_an_mean,
                "zor_orani": 1.0,
                "yari_zor_orani": 0.0,
                "kolay_orani": 0.0
            }
            return kayip, istatistikler

        else: # "batch_all" ve "batch_semi_hard"
            # 3D Tensör Boyutları: (a, p, n) -> (B, B, B)
            d_ap = mesafe_matrisi.unsqueeze(2) # (B, B, 1)
            d_an = mesafe_matrisi.unsqueeze(1) # (B, 1, B)
            
            triplet_kayiplari = d_ap - d_an + self.marjin # (B, B, B)

            triplet_maskesi = (
                pozitif_maske.unsqueeze(2) & # a ve p aynı sınıf
                negatif_maske.unsqueeze(1)   # a ve n farklı sınıf
            )

            # Sınıflandırma Mantığı
            zor_maskesi = triplet_maskesi & (d_an < d_ap)
            yari_zor_maskesi = triplet_maskesi & (d_ap < d_an) & (d_an < d_ap + self.marjin)
            kolay_maskesi = triplet_maskesi & (d_an >= d_ap + self.marjin)

            toplam_gecerli = triplet_maskesi.sum().item()
            zor_sayisi = zor_maskesi.sum().item()
            yari_zor_sayisi = yari_zor_maskesi.sum().item()
            kolay_sayisi = kolay_maskesi.sum().item()

            if strateji == "batch_semi_hard":
                # Yalnızca yarı-zorları veya loss > 0 olanları seç
                hedef_maske = yari_zor_maskesi | zor_maskesi
            else: # batch_all
                hedef_maske = triplet_maskesi & (triplet_kayiplari > 0)

            aktif_kayiplar = triplet_kayiplari[hedef_maske]
            kayip = aktif_kayiplar.mean() if aktif_kayiplar.numel() > 0 else torch.tensor(0.0, device=cihaz, requires_grad=True)

            d_ap_gecerli = d_ap.expand_as(triplet_kayiplari)[triplet_maskesi]
            d_an_gecerli = d_an.expand_as(triplet_kayiplari)[triplet_maskesi]

            d_ap_mean = d_ap_gecerli.mean().item() if d_ap_gecerli.numel() > 0 else 0.0
            d_an_mean = d_an_gecerli.mean().item() if d_an_gecerli.numel() > 0 else 0.0

            istatistikler = {
                "toplam_triplet": toplam_gecerli,
                "aktif_triplet_orani": (len(aktif_kayiplar) / max(1, toplam_gecerli)) * 100.0,
                "d_ap_ort": d_ap_mean,
                "d_an_ort": d_an_mean,
                "zor_orani": (zor_sayisi / max(1, toplam_gecerli)) * 100.0,
                "yari_zor_orani": (yari_zor_sayisi / max(1, toplam_gecerli)) * 100.0,
                "kolay_orani": (kolay_sayisi / max(1, toplam_gecerli)) * 100.0
            }
            return kayip, istatistikler
