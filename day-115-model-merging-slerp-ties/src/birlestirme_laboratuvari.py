"""
Model Birleştirme ve Füzyon Laboratuvarı (Day 115).
Uzman modellerin eğitimi, Linear vs SLERP vs TIES vs DARE-TIES füzyonu ve çok alanlı (Multi-Task) test motoru.
"""

from typing import Dict, Any, List, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ag_mimarisi import UzmanModel
from .model_birlestirici import ModelBirlestirici


class BirlestirmeLaboratuvari:
    """Model Birleştirme ve Çok Alanlı Başarım Laboratuvarı."""

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 32,
        cihaz: torch.device = torch.device("cpu"),
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.cihaz = cihaz

        # 1. Ortak Taban Modeli
        torch.manual_seed(42)
        self.taban_model = UzmanModel(in_dim, hidden_dim, out_dim).to(cihaz)

    def gorev_verisi_uret(
        self,
        gorev_tipi: str = "matematik",
        ornek_sayisi: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Belirli bir alan için sentetik girdi-hedef çiftleri üretir."""
        torch.manual_seed(hash(gorev_tipi) % 100000)
        X = torch.randn(ornek_sayisi, self.in_dim, device=self.cihaz)

        if gorev_tipi == "matematik":
            # Doğrusal olmayan polinom dönüşümü
            W = torch.randn(self.in_dim, self.out_dim, device=self.cihaz) * 0.8
            Y = torch.tanh(X @ W) * 2.0
        elif gorev_tipi == "kodlama":
            # Parçalı mantıksal dönüşüm
            W = torch.randn(self.in_dim, self.out_dim, device=self.cihaz) * 1.2
            Y = torch.sin(X @ W) + torch.relu(X @ W[:, :self.out_dim])
        else:  # genel akıl yürütme
            W = torch.randn(self.in_dim, self.out_dim, device=self.cihaz) * 0.5
            Y = torch.sigmoid(X @ W)

        return X, Y

    def uzman_model_egit(
        self,
        gorev_tipi: str = "matematik",
        epok_sayisi: int = 35,
        lr: float = 1e-2,
    ) -> nn.Module:
        """Ortak taban modelden başlayarak belirli bir alanda uzmanlaşmış model eğitir."""
        model = copy.deepcopy(self.taban_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X, Y = self.gorev_verisi_uret(gorev_tipi, ornek_sayisi=300)

        for _ in range(epok_sayisi):
            model.train()
            pred = model(X)
            loss = criterion(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def model_degerlendir(
        self,
        model: nn.Module,
        gorev_tipi: str = "matematik",
    ) -> float:
        """Modelin belirli bir görevdeki başarısını (0-100 puan) hesaplar."""
        model.eval()
        X_test, Y_test = self.gorev_verisi_uret(gorev_tipi, ornek_sayisi=100)

        with torch.no_grad():
            pred = model(X_test)
            mse = F.mse_loss(pred, Y_test).item()

        # MSE kaybını 0-100 aralığında bir başarı skoruna dönüştür
        skor = max(0.0, min(100.0, 100.0 * (1.0 / (1.0 + mse))))
        return float(skor)

    def fuzyon_deneyini_kostur(self) -> Dict[str, Dict[str, float]]:
        """Tüm birleştirme yöntemlerini uygular ve çok alanlı başarımları raporlar."""
        # 1. Uzman modelleri eğit
        matematik_uzmani = self.uzman_model_egit("matematik", epok_sayisi=35)
        kodlama_uzmani = self.uzman_model_egit("kodlama", epok_sayisi=35)

        # 2. Birleştirme yöntemlerini çalıştır
        lineer_model = ModelBirlestirici.lineer_birlestir(
            self.taban_model, [matematik_uzmani, kodlama_uzmani], agirliklar=[0.5, 0.5]
        )
        slerp_model = ModelBirlestirici.slerp_birlestir(
            matematik_uzmani, kodlama_uzmani, t=0.5
        )
        ties_model = ModelBirlestirici.ties_birlestir(
            self.taban_model, [matematik_uzmani, kodlama_uzmani], agirliklar=[0.5, 0.5], trim_orani=0.3
        )
        dare_ties_model = ModelBirlestirici.dare_birlestir(
            self.taban_model, [matematik_uzmani, kodlama_uzmani], agirliklar=[0.5, 0.5], drop_orani=0.5, ties_uygula=True
        )

        modeller = {
            "Taban Model (Base)": self.taban_model,
            "Matematik Uzmanı": matematik_uzmani,
            "Kodlama Uzmanı": kodlama_uzmani,
            "Linear Merge": lineer_model,
            "SLERP Merge": slerp_model,
            "TIES Merge": ties_model,
            "DARE-TIES Merge": dare_ties_model,
        }

        sonuclar = {}
        for isim, m in modeller.items():
            skor_mat = self.model_degerlendir(m, "matematik")
            skor_kod = self.model_degerlendir(m, "kodlama")
            skor_genel = self.model_degerlendir(m, "genel")
            bilesik = (skor_mat + skor_kod + skor_genel) / 3.0

            sonuclar[isim] = {
                "Matematik Skoru": skor_mat,
                "Kodlama Skoru": skor_kod,
                "Genel Akıl Yürütme": skor_genel,
                "Bileşik Başarı": bilesik,
            }

        return sonuclar
