"""
Ring All-Reduce ve Gradient Bucketing İletişim Motoru (Day 181 - FAZ 10).
Baidu / PyTorch DDP Ring All-Reduce Algoritması ve 25 MB'lık Gradyan Paketleme Mekanizması.
"""

from typing import List, Dict, Any, Tuple
import torch
import numpy as np


class RingAllReduceSimulasyonu:
    """
    Halka Topolojisi Üzerinde Ring All-Reduce İletişim Algoritması.
    - Scatter-Reduce: N-1 adım
    - All-Gather: N-1 adım
    - Toplam Adım: 2*(N-1)
    - İletişim Hacmi: 2 * (N-1)/N * M bayt (GPU sayısından bağımsız O(1) bant genişliği yükü)
    """

    def __init__(self, num_ranks: int = 4):
        assert num_ranks >= 2, "Ring All-Reduce en az 2 rank (GPU) gerektirir."
        self.num_ranks = num_ranks

    def all_reduce(self, rank_tensorleri: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        N adet GPU rank'inin tensörlerini Ring All-Reduce ile toplar ve ortalamasını alır.
        rank_tensorleri: [Tensor_0, Tensor_1, ..., Tensor_{N-1}], her biri aynı boyutta [M]
        """
        assert len(rank_tensorleri) == self.num_ranks, f"Beklenen rank sayısı: {self.num_ranks}, gelen: {len(rank_tensorleri)}"
        N = self.num_ranks

        # Orijinal tensör boyutu M
        flat_tensors = [t.clone().detach().flatten().float() for t in rank_tensorleri]
        M = flat_tensors[0].shape[0]

        # M'nin N'e tam bölünmesini sağla (Padding)
        pad_size = (N - (M % N)) % N
        if pad_size > 0:
            padded_tensors = [torch.cat([t, torch.zeros(pad_size, dtype=t.dtype)]) for t in flat_tensors]
        else:
            padded_tensors = flat_tensors

        M_padded = padded_tensors[0].shape[0]
        chunk_size = M_padded // N

        # Tensörleri N parçaya (chunk) böl: chunks[rank][chunk_idx]
        chunks = [[padded_tensors[r][c * chunk_size:(c + 1) * chunk_size].clone() for c in range(N)] for r in range(N)]

        # -------------------------------------------------------------
        # ADIM 1: SCATTER-REDUCE FAZI (N - 1 Adım)
        # -------------------------------------------------------------
        # Her adım k'da (0..N-2):
        # Rank r, elindeki send_chunk_idx parçasını (r+1)%N'e gönderir.
        # Rank r, (r-1)%N'den recv_chunk_idx parçasını alır ve kendi yerel parçasına toplar.
        for step in range(N - 1):
            next_chunks = [[c.clone() for c in rank_c] for rank_c in chunks]
            for r in range(N):
                recv_idx = (r - step - 1) % N
                prev_rank = (r - 1) % N
                # prev_rank, send_idx = ((r-1) - step) % N = recv_idx parçasını gönderdi
                next_chunks[r][recv_idx] = chunks[r][recv_idx] + chunks[prev_rank][recv_idx]
            chunks = next_chunks

        # Scatter-Reduce sonunda her rank r, tam toplanmış (reduced) bir chunk'a sahiptir: chunk_idx = (r + 1) % N

        # -------------------------------------------------------------
        # ADIM 2: ALL-GATHER FAZI (N - 1 Adım)
        # -------------------------------------------------------------
        # Her rank tam toplanmış parçayı halka boyunca diğer ranklere iletir.
        for step in range(N - 1):
            next_chunks = [[c.clone() for c in rank_c] for rank_c in chunks]
            for r in range(N):
                recv_idx = (r - step) % N
                prev_rank = (r - 1) % N
                # prev_rank, send_idx = ((r-1) - step + 1) % N = recv_idx parçasını gönderdi
                next_chunks[r][recv_idx] = chunks[prev_rank][recv_idx].clone()
            chunks = next_chunks

        # -------------------------------------------------------------
        # PARÇALARI BİRLEŞTİR VE ORTALAMA AL
        # -------------------------------------------------------------
        sonuc_tensorleri = []
        for r in range(N):
            reconstructed = torch.cat(chunks[r], dim=0)
            if pad_size > 0:
                reconstructed = reconstructed[:-pad_size]
            # Global ortalama için N'e böl
            avg_tensor = (reconstructed / N).reshape(rank_tensorleri[r].shape)
            sonuc_tensorleri.append(avg_tensor)

        # İletişim İstatistikleri
        bytes_per_elem = 4  # float32
        transfer_hacmi_bayt = 2 * ((N - 1) / N) * M * bytes_per_elem
        ps_transfer_hacmi_bayt = 2 * (N - 1) * M * bytes_per_elem  # Parameter Server merkezi darboğazı

        istatistikler = {
            "num_ranks": N,
            "tensor_boyutu": M,
            "scatter_reduce_adim": N - 1,
            "all_gather_adim": N - 1,
            "toplam_ring_adimi": 2 * (N - 1),
            "ring_transfer_mb": round(transfer_hacmi_bayt / (1024 * 1024), 3),
            "ps_merkezi_transfer_mb": round(ps_transfer_hacmi_bayt / (1024 * 1024), 3),
            "bant_genisligi_avantaji": f"{N:.1f}x daha verimli (Parameter Server'a kıyasla)",
        }

        return sonuc_tensorleri, istatistikler


class GradyanPaketleyici:
    """
    PyTorch DDP Gradient Bucketing Mekanizması.
    Yüzlerce küçük parametre gradyanını contiguous 25 MB'lık havuzlarda (Bucket) birleştirerek
    IPC ve All-Reduce çağrı overhead'ini 100'lerden 2-3 çağrıya düşürür.
    """

    def __init__(self, bucket_cap_mb: float = 25.0):
        self.bucket_cap_mb = bucket_cap_mb
        self.bucket_cap_bytes = bucket_cap_mb * 1024 * 1024

    def parametreleri_paketle(
        self,
        parametreler: List[torch.nn.Parameter],
    ) -> List[List[torch.nn.Parameter]]:
        """
        Model parametrelerini hedef bucket kapasitesine göre gruplar.
        Geri geçiş sırasına uygunluk için tersten (ters katman sırasıyla) paketleme yapar.
        """
        buckets = []
        current_bucket = []
        current_bytes = 0

        # Geri geçiş son katmandan başladığı için parametreleri ters sırala
        for p in reversed(parametreler):
            if not p.requires_grad:
                continue

            param_bytes = p.numel() * p.element_size()

            if current_bytes + param_bytes > self.bucket_cap_bytes and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [p]
                current_bytes = param_bytes
            else:
                current_bucket.append(p)
                current_bytes += param_bytes

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    @classmethod
    def paket_istatistikleri_hesapla(
        cls,
        parametreler: List[torch.nn.Parameter],
        bucket_cap_mb: float = 25.0,
    ) -> Dict[str, Any]:
        """Gradient Bucketing öncesi ve sonrası iletişim çağrısı kıyaslama analizi."""
        paketleyici = cls(bucket_cap_mb=bucket_cap_mb)
        buckets = paketleyici.parametreleri_paketle(parametreler)

        toplam_param_sayisi = sum(p.numel() for p in parametreler if p.requires_grad)
        toplam_param_tensori = sum(1 for p in parametreler if p.requires_grad)
        toplam_mb = sum(p.numel() * p.element_size() for p in parametreler if p.requires_grad) / (1024 * 1024)

        bucket_sayisi = len(buckets)
        ipc_kazanci = toplam_param_tensori / max(bucket_sayisi, 1)

        return {
            "toplam_parametre_adedi": toplam_param_sayisi,
            "toplam_tensor_sayisi": toplam_param_tensori,
            "toplam_model_mb": round(toplam_mb, 2),
            "bucket_sayisi": bucket_sayisi,
            "bucket_kapasite_mb": bucket_cap_mb,
            "ipc_cagri_azalmasi": f"{ipc_kazanci:.1f}x daha az All-Reduce çağrısı",
            "tahmini_gecikme_iyilesmesi": f"%{max(0, 100 - (100 / ipc_kazanci)):.1f} IPC tasarrufu",
        }
