"""
Sıfırdan Çok Kafalı Öz Dikkat Mekanizması (Multi-Head Self-Attention - MHSA)
----------------------------------------------------------------------------
D_model boyutunu H adet bağımsız dikkat uzayına (Heads) bölen, paralel
tensör matris çarpımlarıyla hesaplayan ve çıkış projeksiyonu uygulayan modül.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .scaled_dot_product import OlcekliNoktaCarpimDikkat


class CokKafaliOzDikkat(nn.Module):
    """
    Çok Kafalı Öz Dikkat (Multi-Head Self-Attention):
    MultiHead(Q, K, V) = Concat(head_1, ..., head_H) * W_O
    """
    def __init__(
        self,
        model_boyutu: int = 64,
        kafa_sayisi: int = 4,
        dropout_orani: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert model_boyutu % kafa_sayisi == 0, (
            f"Model boyutu ({model_boyutu}) kafa sayısına ({kafa_sayisi}) tam bölünmelidir!"
        )

        self.model_boyutu = model_boyutu
        self.kafa_sayisi = kafa_sayisi
        self.d_k = model_boyutu // kafa_sayisi

        # Q, K, V Doğrusal Projeksiyon Katmanları
        self.w_q = nn.Linear(model_boyutu, model_boyutu, bias=bias)
        self.w_k = nn.Linear(model_boyutu, model_boyutu, bias=bias)
        self.w_v = nn.Linear(model_boyutu, model_boyutu, bias=bias)

        # Çıkış Birleştirme Projeksiyonu (W_O)
        self.w_o = nn.Linear(model_boyutu, model_boyutu, bias=bias)

        # Çekirdek Dikkat Modülü
        self.dikkat_cekirdegi = OlcekliNoktaCarpimDikkat(dropout_orani=dropout_orani)

    def _kafalara_bol(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi Şekli: (Batch, Seq_Len, Model_Boyutu)
        Çıktı Şekli: (Batch, Kafa_Sayisi, Seq_Len, d_k)
        """
        b, seq_len, _ = x.size()
        # (B, N, H, d_k) -> (B, H, N, d_k)
        return x.view(b, seq_len, self.kafa_sayisi, self.d_k).transpose(1, 2)

    def _kafalari_birlestir(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi Şekli: (Batch, Kafa_Sayisi, Seq_Len, d_k)
        Çıktı Şekli: (Batch, Seq_Len, Model_Boyutu)
        """
        b, _, seq_len, _ = x.size()
        # (B, H, N, d_k) -> (B, N, H, d_k) -> (B, N, Model_Boyutu)
        return x.transpose(1, 2).contiguous().view(b, seq_len, self.model_boyutu)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Öz Dikkat (Self-Attention) veya Çapraz Dikkat (Cross-Attention) yürütür.
        
        Girdiler:
        - q: (B, N_q, D)
        - k: (B, N_k, D) (Eğer None ise k = q)
        - v: (B, N_k, D) (Eğer None ise v = q)
        - mask: (B, 1, N_q, N_k) veya (1, 1, N_q, N_k)
        
        Çıktılar:
        - cikti: (B, N_q, D)
        - dikkat_haritalari: (B, H, N_q, N_k)
        """
        if k is None:
            k = q
        if v is None:
            v = q

        # 1. Doğrusal Projeksiyonlar: (B, N, D) -> (B, N, D)
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # 2. Çoklu Kafalara Bölme: (B, N, D) -> (B, H, N, d_k)
        q_heads = self._kafalara_bol(q_proj)
        k_heads = self._kafalara_bol(k_proj)
        v_heads = self._kafalara_bol(v_proj)

        # 3. Ölçekli Nokta Çarpım Dikkati
        dikkat_ciktisi, dikkat_haritalari = self.dikkat_cekirdegi(
            q_heads, k_heads, v_heads, mask=mask
        )

        # 4. Kafaları Birleştirme: (B, H, N, d_k) -> (B, N, D)
        birlestirilmis = self._kafalari_birlestir(dikkat_ciktisi)

        # 5. Çıkış Projeksiyonu (W_O): (B, N, D) -> (B, N, D)
        cikti = self.w_o(birlestirilmis)

        return cikti, dikkat_haritalari
