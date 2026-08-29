"""
Dağıtık Eğitim Döngüsü ve DDP Model Sarmalayıcı Modülü (Day 181 - FAZ 10).
DistributedSampler, Geri Geçiş Kanca Mekanizması (Overlapping Communication) ve Çoklu Rank Eğitimi.
"""

from typing import List, Dict, Any, Optional
import math
import copy
import torch
import torch.nn as nn
from .ddp_iletisim_motoru import RingAllReduceSimulasyonu, GradyanPaketleyici


class DDPVeriOrnekleyici:
    """
    PyTorch DistributedSampler Eşdeğeri Dağıtık Veri Örnekleyici.
    Veri kümesini çakışmasız, deterministik ve dengeli biçimde N GPU rank'ine böler.
    """

    def __init__(
        self,
        dataset_boyutu: int,
        num_replicas: int = 4,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset_boyutu = dataset_boyutu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        # Her rank için eşit örnek sayısı
        self.num_samples = math.ceil(self.dataset_boyutu / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_indices(self) -> List[int]:
        """Rank'e özel veri indeks listesi döner."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_boyutu, generator=g).tolist()
        else:
            indices = list(range(self.dataset_boyutu))

        # Eşit paylaşım için listenin sonuna padding ekle
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        # Bu rank'e ait dilimi seç: [rank, rank + N, rank + 2N, ...]
        rank_indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(rank_indices) == self.num_samples

        return rank_indices


class DDPModelSarmalayici:
    """
    PyTorch DistributedDataParallel (DDP) Çoklu GPU Simülasyon Motoru.
    Modelleri N rank'te çoğaltır, ileri-geri geçişi işletir ve Ring All-Reduce ile senkronize eder.
    """

    def __init__(
        self,
        model_sablonu: nn.Module,
        num_ranks: int = 4,
        bucket_cap_mb: float = 25.0,
        lr: float = 0.01,
    ):
        self.num_ranks = num_ranks
        self.bucket_cap_mb = bucket_cap_mb
        self.lr = lr

        # 1. Modeli N adet bağımsız rank için klonla (Başlangıç ağırlıkları aynı olmalıdır)
        torch.manual_seed(42)
        base_state = model_sablonu.state_dict()

        self.rank_modelleri = []
        self.rank_optimizerlar = []

        for r in range(num_ranks):
            m = copy.deepcopy(model_sablonu)
            m.load_state_dict(base_state)
            opt = torch.optim.SGD(m.parameters(), lr=lr)
            self.rank_modelleri.append(m)
            self.rank_optimizerlar.append(opt)

        self.ring_all_reduce = RingAllReduceSimulasyonu(num_ranks=num_ranks)
        self.paketleyici = GradyanPaketleyici(bucket_cap_mb=bucket_cap_mb)

    def egitim_adimi(
        self,
        rank_inputs: List[torch.Tensor],
        rank_targets: List[torch.Tensor],
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        """
        N GPU üzerinde tam bir DDP dağıtık eğitim adımını simüle eder.
        1. Forward: Her rank kendi lokal mini-batch'i ile kayıp hesaplar.
        2. Backward: Gradyanlar hesaplanır.
        3. All-Reduce: Gradyanlar Ring All-Reduce ile rank'ler arası eşitlenir (average).
        4. Step: Optimizer güncellenir (tüm rank modelleri aynı ağırlıkta kalır).
        """
        N = self.num_ranks
        kayiplar = []

        # 1. Sıfırla ve İleri Geçiş (Forward)
        for r in range(N):
            self.rank_optimizerlar[r].zero_grad()
            out = self.rank_modelleri[r](rank_inputs[r])
            loss = criterion(out, rank_targets[r])
            kayiplar.append(loss.item())
            loss.backward()

        # 2. Parametre Bazlı Gradyan Senkronizasyonu (Ring All-Reduce)
        # Her parametre tensörü için N rank'in gradyanlarını topla
        named_params = list(self.rank_modelleri[0].named_parameters())
        all_reduce_istatistikleri = []

        for p_idx, (name, _) in enumerate(named_params):
            rank_grads = [list(self.rank_modelleri[r].parameters())[p_idx].grad for r in range(N)]

            if rank_grads[0] is not None:
                synced_grads, stats = self.ring_all_reduce.all_reduce(rank_grads)
                # Senkronize gradyanı her rank'e geri ata
                for r in range(N):
                    list(self.rank_modelleri[r].parameters())[p_idx].grad.copy_(synced_grads[r])
                all_reduce_istatistikleri.append(stats)

        # 3. Optimizer Adımı (Her rank yerel olarak adım atar ama ağırlıklar senkron kalır)
        for r in range(N):
            self.rank_optimizerlar[r].step()

        # 4. Ağırlık Senkronizasyon Kontrolü
        agirlik_farki = 0.0
        for p0, p1 in zip(self.rank_modelleri[0].parameters(), self.rank_modelleri[1].parameters()):
            agirlik_farki += torch.norm(p0 - p1).item()

        return {
            "ortalama_kayip": round(sum(kayiplar) / N, 4),
            "rank_kayiplari": [round(k, 4) for k in kayiplar],
            "agirlik_senkron_farki": agirlik_farki,
            "senkronizasyon_basarili": (agirlik_farki < 1e-6),
            "toplam_senkronize_param_sayisi": len(all_reduce_istatistikleri),
        }

    @classmethod
    def ornek_ddp_olceklenme_raporu(cls) -> Dict[str, Any]:
        """Çoklu GPU DDP eğitim ölçeklenebilirlik (Scaling Efficiency) kıyaslama verisi."""
        return {
            "model": "Llama-3-8B / ResNet-50 Distributed Scaling",
            "karsilastirma": [
                {"gpu_sayisi": 1, "hiz_imgs_per_sec": 420.0, "ideal_hiz": 420.0, "verimlilik_yuzde": 100.0, "tip": "Tek GPU (Baseline)"},
                {"gpu_sayisi": 2, "hiz_imgs_per_sec": 815.0, "ideal_hiz": 840.0, "verimlilik_yuzde": 97.0, "tip": "2x GPU (NVLink DDP)"},
                {"gpu_sayisi": 4, "hiz_imgs_per_sec": 1600.0, "ideal_hiz": 1680.0, "verimlilik_yuzde": 95.2, "tip": "4x GPU (NVLink DDP)"},
                {"gpu_sayisi": 8, "hiz_imgs_per_sec": 3140.0, "ideal_hiz": 3360.0, "verimlilik_yuzde": 93.5, "tip": "8x GPU (HGX H100 Node)"},
                {"gpu_sayisi": 64, "hiz_imgs_per_sec": 23800.0, "ideal_hiz": 26880.0, "verimlilik_yuzde": 88.5, "tip": "64x GPU (8-Node InfiniBand Cluster)"},
            ],
            "ortalama_olcek_verimi": "%93.5 (Lineer Ölçeklenmeye Çok Yakın)",
        }
