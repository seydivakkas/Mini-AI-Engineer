"""
Model Birleştirme (Model Merging) Çekirdek Algoritmaları Modülü (Day 115).
SLERP, Linear (Task Arithmetic), TIES-Merging ve DARE (Drop And REscale) teknikleri.
Sıfır GPU eğitimi ve geriye yayılım olmadan parametre uzayı füzyonu.
"""

from typing import List, Dict, Tuple, Optional
import copy
import torch
import torch.nn as nn


def slerp_tensor(v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """
    İki tensör arasında küresel doğrusal enterpolasyon (SLERP) uygular.
    v0, v1: Aynı şekle sahip parametre tensörleri.
    t: Ağırlık faktörü [0.0, 1.0] (0.0 -> v0, 1.0 -> v1)
    """
    v0_shape = v0.shape
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()

    norm_v0 = torch.norm(v0_flat)
    norm_v1 = torch.norm(v1_flat)

    if norm_v0 < eps or norm_v1 < eps:
        # Sıfıra yakın norm durumunda lineer enterpolasyon
        return (1.0 - t) * v0 + t * v1

    u0 = v0_flat / norm_v0
    u1 = v1_flat / norm_v1

    dot = torch.sum(u0 * u1).clamp(-1.0 + eps, 1.0 - eps)

    # Vektörler neredeyse paralel ise lineer enterpolasyon yap
    if abs(float(dot.item())) > 0.9995:
        sonuc_flat = (1.0 - t) * v0_flat + t * v1_flat
        return sonuc_flat.view(v0_shape)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    scale0 = torch.sin((1.0 - t) * omega) / sin_omega
    scale1 = torch.sin(t * omega) / sin_omega

    sonuc_flat = scale0 * v0_flat + scale1 * v1_flat
    return sonuc_flat.view(v0_shape)


class ModelBirlestirici:
    """Model Birleştirme (Merging) Motoru."""

    @staticmethod
    def lineer_birlestir(
        taban_model: nn.Module,
        modeller: List[nn.Module],
        agirliklar: Optional[List[float]] = None,
    ) -> nn.Module:
        """
        Task Arithmetic (Görev Aritmetiği) doğrusal birleştirme:
        theta_merged = theta_base + sum_i (alpha_i * (theta_i - theta_base))
        """
        if agirliklar is None:
            agirliklar = [1.0 / len(modeller)] * len(modeller)

        birlesmis_model = copy.deepcopy(taban_model)
        taban_state = taban_model.state_dict()
        birlesmis_state = copy.deepcopy(taban_state)

        for isim in taban_state.keys():
            if not taban_state[isim].is_floating_point():
                continue
            delta_toplam = torch.zeros_like(taban_state[isim])
            for model, alpha in zip(modeller, agirliklar):
                m_state = model.state_dict()
                tau = m_state[isim] - taban_state[isim]
                delta_toplam += alpha * tau
            birlesmis_state[isim] = taban_state[isim] + delta_toplam

        birlesmis_model.load_state_dict(birlesmis_state)
        return birlesmis_model

    @staticmethod
    def slerp_birlestir(
        model_a: nn.Module,
        model_b: nn.Module,
        t: float = 0.5,
    ) -> nn.Module:
        """
        İki modelin tüm ağırlıklarını küresel yüzeyde (SLERP) enterpole eder.
        """
        birlesmis_model = copy.deepcopy(model_a)
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        birlesmis_state = {}

        for isim in state_a.keys():
            param_a = state_a[isim]
            param_b = state_b[isim]
            if param_a.is_floating_point():
                birlesmis_state[isim] = slerp_tensor(param_a, param_b, t=t)
            else:
                birlesmis_state[isim] = param_a

        birlesmis_model.load_state_dict(birlesmis_state)
        return birlesmis_model

    @staticmethod
    def ties_birlestir(
        taban_model: nn.Module,
        modeller: List[nn.Module],
        agirliklar: Optional[List[float]] = None,
        trim_orani: float = 0.5,
    ) -> nn.Module:
        """
        TIES-Merging (TRIM, ELECT SIGN & MERGE - Yadav et al.):
        1. Trim: En yüksek % (1 - trim_orani) mutlak değerli parametreleri tut, gerisini sıfırla.
        2. Elect Sign: Parametre bazında mutabakat işaretini belirle s_j = sgn(sum tau_ij), çelişenleri sıfırla.
        3. Disjoint Merge: Hayatta kalan değerlerin ortalamasını al.
        """
        if agirliklar is None:
            agirliklar = [1.0 / len(modeller)] * len(modeller)

        birlesmis_model = copy.deepcopy(taban_model)
        taban_state = taban_model.state_dict()
        birlesmis_state = copy.deepcopy(taban_state)

        for isim in taban_state.keys():
            if not taban_state[isim].is_floating_point():
                continue

            # Görev vektörlerini (delta) hesapla
            tau_list = []
            for model in modeller:
                tau = model.state_dict()[isim] - taban_state[isim]
                # 1. TRIM ADIMI (Düşük genlikli gürültüyü buda)
                if trim_orani > 0.0:
                    k = int(tau.numel() * (1.0 - trim_orani))
                    if k > 0 and k < tau.numel():
                        esik = torch.topk(tau.abs().flatten(), k).values[-1]
                        tau = torch.where(tau.abs() >= esik, tau, torch.zeros_like(tau))
                tau_list.append(tau)

            # 2. ELECT SIGN ADIMI (İşaret Mutabakatı)
            tau_stack = torch.stack(tau_list, dim=0)  # [M, ...]
            isaret_toplam = tau_stack.sum(dim=0)
            mutabakat_isareti = torch.sign(isaret_toplam)
            mutabakat_isareti[mutabakat_isareti == 0] = 1.0

            # Mutabakatla uyuşmayanları sıfırla
            uyusan_tau = torch.where(torch.sign(tau_stack) == mutabakat_isareti, tau_stack, torch.zeros_like(tau_stack))

            # 3. DISJOINT MERGE (Ağırlıklı Ortalama)
            agirlik_tensor = torch.tensor(agirliklar, device=taban_state[isim].device, dtype=taban_state[isim].dtype)
            for _ in range(tau_stack.dim() - 1):
                agirlik_tensor = agirlik_tensor.unsqueeze(-1)

            agirlikli_toplam = (uyusan_tau * agirlik_tensor).sum(dim=0)
            birlesmis_state[isim] = taban_state[isim] + agirlikli_toplam

        birlesmis_model.load_state_dict(birlesmis_state)
        return birlesmis_model

    @staticmethod
    def dare_birlestir(
        taban_model: nn.Module,
        modeller: List[nn.Module],
        agirliklar: Optional[List[float]] = None,
        drop_orani: float = 0.5,
        ties_uygula: bool = True,
    ) -> nn.Module:
        """
        DARE (Drop And REscale - Yu et al.):
        Görev vektörü parametrelerini p olasılıkla atıp (Drop), kalanları 1/(1-p) ile yeniden ölçekler (Rescale).
        Ardından TIES veya lineer birleştirme ile birleştirir.
        """
        if agirliklar is None:
            agirliklar = [1.0 / len(modeller)] * len(modeller)

        dare_modeller = []
        taban_state = taban_model.state_dict()

        for model in modeller:
            m_kopyasi = copy.deepcopy(model)
            m_state = m_kopyasi.state_dict()

            for isim in taban_state.keys():
                if not taban_state[isim].is_floating_point():
                    continue
                tau = m_state[isim] - taban_state[isim]
                # DARE: Bernoulli maskesi ve yeniden ölçekleme
                if drop_orani > 0.0 and drop_orani < 1.0:
                    keep_prob = 1.0 - drop_orani
                    mask = (torch.rand_like(tau) < keep_prob).to(tau.dtype)
                    tau_dare = (mask * tau) / keep_prob
                else:
                    tau_dare = tau
                m_state[isim] = taban_state[isim] + tau_dare

            m_kopyasi.load_state_dict(m_state)
            dare_modeller.append(m_kopyasi)

        if ties_uygula:
            return ModelBirlestirici.ties_birlestir(taban_model, dare_modeller, agirliklar=agirliklar, trim_orani=0.0)
        else:
            return ModelBirlestirici.lineer_birlestir(taban_model, dare_modeller, agirliklar=agirliklar)
