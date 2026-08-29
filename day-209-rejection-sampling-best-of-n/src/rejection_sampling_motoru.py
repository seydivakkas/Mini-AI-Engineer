"""
Rejection Sampling & Best-of-N Motoru (Day 209 - FAZ 11).
Sıcaklık Örneklemesi, Doğrulayıcı Filtreleme ve Sentetik SFT Eğitimi.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicySampler:
    """
    Sıcaklık (Temperature) Tabanlı Çoklu Aday Üretim Motoru.
    Verilen bir prompt için K adet bağımsız aday düşünce yolu üretir.
    """

    @classmethod
    def orneklem_uret(
        cls,
        prompt: str,
        k_orneklem: int = 8,
        sicaklik: float = 0.8,
        taban_dogruluk_olasiligi: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Sıcaklık parametresine göre çeşitlilik ve kalite dengesini modelleyen aday üretimi.
        Düşük T (0.2) -> Düşük çeşitlilik, ortalama kalite
        Optimal T (0.8) -> Yüksek çeşitlilik, kaliteli çözüm yakalama şansı yüksek
        Yüksek T (1.5) -> Çok yüksek gürültü/halüsinasyon
        """
        adaylar = []
        for i in range(k_orneklem):
            # Sıcaklık etkisi: T=0.8 civarında en yüksek keşif başarısı
            if sicaklik < 0.3:
                basari_p = taban_dogruluk_olasiligi * 1.05
                gurultu = random.gauss(0, 0.05)
            elif sicaklik <= 0.9:
                basari_p = taban_dogruluk_olasiligi * (1.0 + (sicaklik - 0.2) * 0.4)
                gurultu = random.gauss(0, 0.12)
            else:
                basari_p = taban_dogruluk_olasiligi * max(0.2, 1.4 - sicaklik * 0.5)
                gurultu = random.gauss(0, 0.30)

            eff_p = min(max(basari_p + gurultu, 0.05), 0.98)
            dogru_mu = random.random() < eff_p
            kalite_skoru = (0.75 + random.uniform(0.1, 0.25)) if dogru_mu else random.uniform(0.0, 0.45)

            adaylar.append({
                "aday_id": i + 1,
                "prompt": prompt,
                "yanit": f"Cozum_Yolu_{i+1}_(T={sicaklik})_Sonuc: {'DOGRU' if dogru_mu else 'YANLIS'}",
                "dogru_mu": dogru_mu,
                "odul_skoru": float(kalite_skoru),
                "sicaklik": sicaklik,
            })
        return adaylar


class RejectionFilter:
    """
    Doğrulayıcı ve Ödül Eşiğine Göre Hatalı Yolları Eleyen Filtreleme Motoru.
    """

    @classmethod
    def adaylari_filtrele_ve_sec(
        cls,
        adaylar: List[Dict[str, Any]],
        esik_skoru: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Adayları filtreler ve Best-of-K kazananını seçer.
        """
        if not adaylar:
            raise ValueError("Aday listesi boş olamaz.")

        kabul_edilenler = [a for a in adaylar if a["odul_skoru"] >= esik_skoru]
        reddedilenler = [a for a in adaylar if a["odul_skoru"] < esik_skoru]

        kabul_orani = len(kabul_edilenler) / len(adaylar)

        # En iyi adayı seç
        en_iyi_aday = max(adaylar, key=lambda x: x["odul_skoru"])
        en_iyi_kabul_mu = en_iyi_aday["odul_skoru"] >= esik_skoru

        return {
            "toplam_aday": len(adaylar),
            "kabul_sayisi": len(kabul_edilenler),
            "red_sayisi": len(reddedilenler),
            "kabul_orani": kabul_orani,
            "en_iyi_aday": en_iyi_aday,
            "en_iyi_kabul_mu": en_iyi_kabul_mu,
            "kabul_edilenler": kabul_edilenler,
        }


class RSSFTDatasetBuilder:
    """
    Rejection Sampling ile Sentetik Yüksek Kaliteli SFT Veri Kümesi İnşa Edici.
    """

    @classmethod
    def sentetik_veri_seti_olustur(
        cls,
        problem_havuzu: List[str],
        k_orneklem: int = 8,
        sicaklik: float = 0.8,
        esik_skoru: float = 0.60,
    ) -> Dict[str, Any]:
        """Tüm problemler için örnekleme ve filtreleme yaparak SFT çiftleri üretir."""
        sft_ornekleri = []
        toplam_uretilen = 0
        basarili_problem_sayisi = 0

        for prompt in problem_havuzu:
            adaylar = PolicySampler.orneklem_uret(prompt, k_orneklem, sicaklik)
            toplam_uretilen += len(adaylar)

            filtre_sonucu = RejectionFilter.adaylari_filtrele_ve_sec(adaylar, esik_skoru)
            if filtre_sonucu["en_iyi_kabul_mu"]:
                sft_ornekleri.append({
                    "prompt": prompt,
                    "hedef_yanit": filtre_sonucu["en_iyi_aday"]["yanit"],
                    "kalite_skoru": filtre_sonucu["en_iyi_aday"]["odul_skoru"],
                })
                basarili_problem_sayisi += 1

        genel_kabul_orani = len(sft_ornekleri) / len(problem_havuzu) if problem_havuzu else 0.0

        return {
            "problem_sayisi": len(problem_havuzu),
            "sft_ornek_sayisi": len(sft_ornekleri),
            "toplam_uretilen_aday": toplam_uretilen,
            "problem_kapsama_orani": genel_kabul_orani,
            "sft_veri_kumesi": sft_ornekleri,
        }


class SimplePolicyModel(nn.Module):
    """SFT Eğitimi için Basit Dil Politikası Modeli."""

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=128, batch_first=True),
            num_layers=2,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        h = self.encoder(x)
        return self.lm_head(h)


class RSSFTTrainer:
    """
    Filtrelenmiş Rejection Sampling Veri Kümesi Üzerinde SFT Kayıp Eğiticisi.
    """

    def __init__(self, model: Optional[SimplePolicyModel] = None, lr: float = 1e-3):
        self.model = model if model is not None else SimplePolicyModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def egitim_adimi(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> Dict[str, float]:
        """SFT Cross-Entropy kaybı hesaplar ve ağırlıkları günceller."""
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(input_ids)
        # [batch * seq, vocab]
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss.backward()
        self.optimizer.step()

        return {"sft_loss": float(loss.item()), "perplexity": float(math.exp(min(loss.item(), 20.0)))}
