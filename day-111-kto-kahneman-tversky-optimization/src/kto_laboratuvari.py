"""
KTO Hizalama ve Davranışsal İktisat Laboratuvarı (Day 111).
Tekil ikili veri üretimi (unpaired binary feedback), KTO eğitim döngüsü ve Beklenti Teorisi analizi.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

from .kto_kaybi import KTOLoss
from .kto_modeli import KTODilModeli


class KTOLaboratuvari:
    """KTO Eğitim, Örtük Ödül Takip ve Beklenti Teorisi Laboratuvarı."""

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        cihaz: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.cihaz = cihaz

        # 1. Politika Modeli (Eğitilebilir pi_theta)
        self.policy_model = KTODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)

        # 2. Referans Model (Dondurulmuş pi_ref)
        self.ref_model = KTODilModeli(
            vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers
        ).to(cihaz)
        self.ref_model.load_state_dict(self.policy_model.state_dict())
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.loss_fn = KTOLoss(beta=0.1, lambda_d=1.0, lambda_u=1.33)

    def tekil_ikili_veri_uret(
        self,
        ornek_sayisi: int = 400,
        prompt_len: int = 10,
        resp_len: int = 14,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        KTO için eşleştirilmemiş (unpaired) tekil ikili veri üretir.
        Her örnek (x, y) ve etiket (+1: Beğenildi, -1: Beğenilmedi) içerir.
        Çıktı: (input_ids, maske, etiketler)
        """
        yari = ornek_sayisi // 2

        v_p_max = max(2, int(self.vocab_size * 0.2))
        v_d_min = v_p_max
        v_d_max = max(v_d_min + 1, int(self.vocab_size * 0.55))
        v_u_min = max(v_d_max, int(self.vocab_size * 0.6))
        v_u_max = self.vocab_size

        # 1. Beğenilen Örnekler (Desirable, +1)
        p_d = torch.randint(1, v_p_max, (yari, prompt_len), device=self.cihaz)
        r_d = torch.randint(v_d_min, v_d_max, (yari, resp_len), device=self.cihaz)
        ids_d = torch.cat([p_d, r_d], dim=1)
        lbl_d = torch.ones(yari, device=self.cihaz)

        # 2. Beğenilmeyen Örnekler (Undesirable, -1) - Farklı rastgele promptlar!
        p_u = torch.randint(1, v_p_max, (yari, prompt_len), device=self.cihaz)
        r_u = torch.randint(v_u_min, v_u_max, (yari, resp_len), device=self.cihaz)
        ids_u = torch.cat([p_u, r_u], dim=1)
        lbl_u = -torch.ones(yari, device=self.cihaz)

        # Birleştir ve karıştır
        all_ids = torch.cat([ids_d, ids_u], dim=0)
        all_lbl = torch.cat([lbl_d, lbl_u], dim=0)

        all_mask = torch.zeros_like(all_ids, dtype=torch.float32)
        all_mask[:, prompt_len:] = 1.0

        perm = torch.randperm(ornek_sayisi)
        return all_ids[perm], all_mask[perm], all_lbl[perm]

    def kto_egit(
        self,
        input_ids: torch.Tensor,
        maske: torch.Tensor,
        etiketler: torch.Tensor,
        epok_sayisi: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 0.1,
    ) -> Dict[str, List[float]]:
        """KTO eğitim döngüsü ve metrik kaydı."""
        self.loss_fn.beta = beta
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)

        N = input_ids.shape[0]
        rapor = {
            "toplam_kayiplar": [],
            "kayiplar_d": [],
            "kayiplar_u": [],
            "dogruluklar": [],
            "r_d_ort": [],
            "r_u_ort": [],
            "marjinler": [],
        }

        for ep in range(epok_sayisi):
            self.policy_model.train()
            perm = torch.randperm(N)
            ep_loss, ep_ld, ep_lu, ep_acc, ep_rd, ep_ru, ep_margin = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            adim_sayisi = 0

            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                b_ids, b_mask, b_lbl = input_ids[idx], maske[idx], etiketler[idx]

                # Politika modeli log-olasılıkları
                pi_logps = self.policy_model.logprob_hesapla(b_ids, b_mask)

                # Dondurulmuş referans model log-olasılıkları
                with torch.no_grad():
                    ref_logps = self.ref_model.logprob_hesapla(b_ids, b_mask)

                loss, metrikler = self.loss_fn(pi_logps, ref_logps, b_lbl)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                optimizer.step()

                ep_loss += float(metrikler["toplam_kayip"].item())
                ep_ld += float(metrikler["kayip_d"].item())
                ep_lu += float(metrikler["kayip_u"].item())
                ep_acc += float(metrikler["dogruluk"].item())
                ep_rd += float(metrikler["ortuk_odul_d"].item())
                ep_ru += float(metrikler["ortuk_odul_u"].item())
                ep_margin += float(metrikler["marjin"].item())
                adim_sayisi += 1

            rapor["toplam_kayiplar"].append(ep_loss / adim_sayisi)
            rapor["kayiplar_d"].append(ep_ld / adim_sayisi)
            rapor["kayiplar_u"].append(ep_lu / adim_sayisi)
            rapor["dogruluklar"].append((ep_acc / adim_sayisi) * 100.0)
            rapor["r_d_ort"].append(ep_rd / adim_sayisi)
            rapor["r_u_ort"].append(ep_ru / adim_sayisi)
            rapor["marjinler"].append(ep_margin / adim_sayisi)

        return rapor
