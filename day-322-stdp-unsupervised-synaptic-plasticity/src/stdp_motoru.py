"""
Day 322: Spike-Timing-Dependent Plasticity (STDP) & Unsupervised Learning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Spike-Timing-Dependent Plasticity (STDP) Hebbian biyolojik öğrenme kuralını,
iz (trace) güncelleme dinamiğini, Winner-Take-All (WTA) yanal inhibisyon mekanizmasını
ve denetimsiz (unsupervised) spiking sinir ağı katmanını içerir.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class STDPLearningRule(nn.Module):
    """
    İz Tabanlı (Trace-based) Çevrimiçi STDP Öğrenme Kuralı.

    Matematiksel Dinamikler:
        Presinaptik İz:
            x_j[t] = beta_pre * x_j[t-1] + S_pre,j[t]
        Postsinaptik İz:
            y_i[t] = beta_post * y_i[t-1] + S_post,i[t]

        Ağırlık Değişimi:
            Delta W_ij = A_plus * S_post,i[t] * x_j[t]  (LTP - Güçlenme)
                       - A_minus * S_pre,j[t] * y_i[t] (LTD - Zayıflama)

        Güncelleme:
            W_ij <- clamp(W_ij + eta * Delta W_ij, w_min, w_max)
    """
    def __init__(
        self,
        a_plus: float = 0.02,
        a_minus: float = 0.015,
        beta_pre: float = 0.9,
        beta_post: float = 0.9,
        learning_rate: float = 0.01,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        super().__init__()
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.beta_pre = beta_pre
        self.beta_post = beta_post
        self.lr = learning_rate
        self.w_min = w_min
        self.w_max = w_max

    def init_traces(
        self, batch_size: int, in_features: int, out_features: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Presinaptik ve postsinaptik iz matrislerini 0 ile başlatır.
        """
        trace_pre = torch.zeros((batch_size, in_features), device=device)
        trace_post = torch.zeros((batch_size, out_features), device=device)
        return trace_pre, trace_post

    def update_weights(
        self,
        weights: torch.Tensor,
        s_pre: torch.Tensor,
        s_post: torch.Tensor,
        trace_pre: torch.Tensor,
        trace_post: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tek bir t zaman adımı için STDP ağırlık ve iz güncellemesi.

        Girdiler:
            weights: (Out_Features, In_Features)
            s_pre: (Batch, In_Features)
            s_post: (Batch, Out_Features)
            trace_pre: (Batch, In_Features)
            trace_post: (Batch, Out_Features)
        """
        # 1. İz Güncellemeleri
        new_trace_pre = self.beta_pre * trace_pre + s_pre
        new_trace_post = self.beta_post * trace_post + s_post

        # 2. LTP (Long-Term Potentiation): S_post * Trace_pre
        # Presinaptik spike geldikten sonra postsinaptik spike oluşursa ağırlık artar
        ltp = torch.bmm(s_post.unsqueeze(2), new_trace_pre.unsqueeze(1))  # (Batch, Out, In)
        
        # 3. LTD (Long-Term Depression): S_pre * Trace_post
        # Postsinaptik spike geldikten sonra presinaptik spike oluşursa ağırlık azalır
        ltd = torch.bmm(new_trace_post.unsqueeze(2), s_pre.unsqueeze(1))  # (Batch, Out, In)

        delta_w_batch = self.a_plus * ltp - self.a_minus * ltd
        
        # Batch üzerinden ortalama ağırlık değişimi
        delta_w = torch.mean(delta_w_batch, dim=0)

        # 4. Ağırlık Güncellemesi ve Kırpma (Clamping)
        new_weights = torch.clamp(weights + self.lr * delta_w, self.w_min, self.w_max)

        return new_weights, new_trace_pre, new_trace_post, delta_w


class WTALateralInhibition(nn.Module):
    """
    Winner-Take-All (WTA - Kazanan Hepsini Alır) Yanal İnhibisyon Mekanizması.
    Aynı katmandaki nöronlar arasında rekabet oluşturarak bir nöronun diğerlerini bastırmasını sağlar.
    """
    def __init__(self, inhibition_strength: float = 1.0):
        super().__init__()
        self.inhibition_strength = inhibition_strength

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Girdi: spikes (Batch, Out_Features)
        Eğer batch örneğinde en az 1 nöron spike vermişse, ilk ateşleyen nöron dışındakileri sıfırlar.
        """
        batch_size, out_features = spikes.shape
        if out_features <= 1:
            return spikes

        # Her batch örneği için ilk spike veren nöronu seç (Winner)
        wta_spikes = spikes.clone()
        has_spikes = (torch.sum(spikes, dim=1, keepdim=True) > 0)
        
        # En yüksek potansiyele/aktivasyona sahip ilk nöronu koru
        winner_indices = torch.argmax(spikes, dim=1, keepdim=True)
        winner_mask = torch.zeros_like(spikes).scatter_(1, winner_indices, 1.0)
        
        wta_spikes = torch.where(has_spikes, spikes * winner_mask, spikes)
        return wta_spikes


class STDPUnsupervisedNetwork(nn.Module):
    """
    Denetimsiz (Unsupervised) STDP Öğrenme Ağı.
    Etiketsiz verilerden Hebbian STDP kuralı ile özellik kümeleme öğrenir.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        time_steps: int = 40,
        beta_neuron: float = 0.85,
        v_threshold: float = 1.0,
        stdp_lr: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_steps = time_steps
        self.beta_neuron = beta_neuron
        self.v_threshold = v_threshold

        # Rastgele pozitif sinaptik ağırlıklar [0.2, 0.5]
        weights_init = torch.rand(out_features, in_features) * 0.3 + 0.2
        self.weights = nn.Parameter(weights_init, requires_grad=False)

        self.stdp_rule = STDPLearningRule(learning_rate=stdp_lr)
        self.wta = WTALateralInhibition()

    def forward(
        self, input_spikes_seq: torch.Tensor, train_stdp: bool = True
    ) -> Dict[str, Any]:
        """
        Girdi: (Batch, Time_Steps, In_Features)
        Çıktı: Sözlük (Spike dizileri, ağırlık değişim geçmişi, izler)
        """
        batch_size, time_steps, _ = input_spikes_seq.shape
        device = input_spikes_seq.device

        # LIF Nöron durumu: V_mem, Refractory Counter
        v_mem = torch.zeros((batch_size, self.out_features), device=device)
        ref_cnt = torch.zeros((batch_size, self.out_features), dtype=torch.int32, device=device)

        trace_pre, trace_post = self.stdp_rule.init_traces(batch_size, self.in_features, self.out_features, device)

        spikes_list = []
        mem_list = []
        delta_w_list = []
        weights_history = [self.weights.clone().detach()]

        for t in range(time_steps):
            s_pre_t = input_spikes_seq[:, t, :]  # (Batch, In)
            
            # Sinaptik Akım: I(t) = S_pre(t) * W^T
            i_t = torch.matmul(s_pre_t, self.weights.t())  # (Batch, Out)

            # LIF Nöron Güncellemesi
            is_refractory = (ref_cnt > 0)
            v_decayed = self.beta_neuron * v_mem + i_t
            v_next = torch.where(is_refractory, 0.0, v_decayed)
            
            raw_spikes_t = (v_next >= self.v_threshold).float()
            
            # Winner-Take-All Yanal İnhibisyon
            s_post_t = self.wta(raw_spikes_t)

            # Reset & Refrakter
            v_mem = torch.where(s_post_t > 0.5, 0.0, v_next)
            ref_cnt = torch.where(s_post_t > 0.5, 2, torch.clamp(ref_cnt - 1, min=0))

            # STDP Ağırlık Güncellemesi
            if train_stdp:
                updated_w, trace_pre, trace_post, delta_w = self.stdp_rule.update_weights(
                    self.weights.data, s_pre_t, s_post_t, trace_pre, trace_post
                )
                self.weights.data = updated_w
                delta_w_list.append(delta_w)

            spikes_list.append(s_post_t)
            mem_list.append(v_mem)

        spikes_seq = torch.stack(spikes_list, dim=1)  # (Batch, T, Out)
        mem_seq = torch.stack(mem_list, dim=1)        # (Batch, T, Out)

        return {
            "spikes_seq": spikes_seq,
            "mem_seq": mem_seq,
            "final_weights": self.weights.clone().detach(),
            "delta_w_list": delta_w_list,
            "trace_pre": trace_pre,
            "trace_post": trace_post,
        }
