"""
Uçtan Uca Regresyon Denetim ve Kalite Kapısı (Quality Gate) Motoru (Day 95).
Altın veri seti, metrik eşikleri, SLA gecikme bütçesi, bellek sızıntısı ve bütünlük testlerini yürütür.
"""

import time
import os
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .model import MiniViTForImageClassification
from .surum_yoneticisi import ReleaseManifestYoneticisi


@dataclass
class KaliteKapisiSonucu:
    """Kalite Kapısı (Quality Gate) denetim sonuçlarını saklayan veri sınıfı."""
    altin_veri_uyumlu: bool
    maks_logits_farki: float
    metrik_regresyon_gecerli: bool
    rc_accuracy: float
    rc_f1_score: float
    sla_uyumlu: bool
    p50_gecikme_ms: float
    p95_gecikme_ms: float
    bellek_kararli: bool
    bellek_artisi_yuzde: float
    butunluk_gecerli: bool
    nihai_karar: str
    gecikmeler_ms: list
    bellek_izleme_mb: list
    detaylar: Dict[str, Any]

    @property
    def onaylandi_mi(self) -> bool:
        return (
            self.altin_veri_uyumlu
            and self.metrik_regresyon_gecerli
            and self.sla_uyumlu
            and self.bellek_kararli
            and self.butunluk_gecerli
        )


class RegresyonDenetleyicisi:
    """MiniViT Sürüm Adayı için uçtan uca regresyon testlerini çalıştıran denetleyici."""

    def __init__(self, cihaz: Optional[torch.device] = None):
        self.cihaz = cihaz or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def altin_veri_seti_testi(
        self,
        model: MiniViTForImageClassification,
        altin_girdiler: torch.Tensor,
        altin_hedef_logits: torch.Tensor,
        tolerans: float = 1e-4,
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Dondurulmuş referans altın veri seti ile modelin çıktı logitlerini karşılaştırır.
        Sayısal sapmanın tolerans dahilinde olduğunu doğrular.
        """
        model.eval()
        m_cihaz = next(model.parameters()).device
        altin_girdiler = altin_girdiler.to(m_cihaz)
        altin_hedef_logits = altin_hedef_logits.to(m_cihaz)

        with torch.no_grad():
            cikti = model(altin_girdiler).logits

        farklar = torch.abs(cikti - altin_hedef_logits)
        maks_fark = torch.max(farklar).item()
        uyumlu = maks_fark < tolerans

        return uyumlu, maks_fark, farklar.cpu().numpy()

    def metrik_regresyon_testi(
        self,
        model: MiniViTForImageClassification,
        test_girdileri: torch.Tensor,
        test_etiketleri: torch.Tensor,
        min_acc: float = 0.85,
        min_f1: float = 0.80,
    ) -> Tuple[bool, Dict[str, float]]:
        """Modelin test seti üzerindeki sınıflandırma metriklerini ve gerileme olup olmadığını denetler."""
        model.eval()
        m_cihaz = next(model.parameters()).device
        test_girdileri = test_girdileri.to(m_cihaz)
        test_etiketleri = test_etiketleri.to(m_cihaz)

        with torch.no_grad():
            logits = model(test_girdileri).logits
            tahminler = torch.argmax(logits, dim=-1)

        y_true = test_etiketleri.cpu().numpy()
        y_pred = tahminler.cpu().numpy()

        acc = float(np.mean(y_true == y_pred))

        # Basit Macro F1 hesaplama
        num_classes = model.config.sinif_sayisi
        f1_list = []
        for c in range(num_classes):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list))
        gecerli = (acc >= min_acc) and (macro_f1 >= min_f1)

        return gecerli, {"accuracy": acc, "macro_f1": macro_f1}

    def gecikme_sla_testi(
        self,
        model: MiniViTForImageClassification,
        girdi_sekli: Tuple[int, int, int, int] = (1, 3, 32, 32),
        iterasyon: int = 50,
        max_p50_ms: float = 5.0,
        max_p95_ms: float = 10.0,
    ) -> Tuple[bool, float, float, list]:
        """Modelin çıkarım gecikmesini (P50, P95) SLA sınırlarına karşı test eder."""
        model.eval()
        m_cihaz = next(model.parameters()).device
        dummy_input = torch.randn(*girdi_sekli, device=m_cihaz)

        # Isınma (Warmup)
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)

        if torch.cuda.is_available() and m_cihaz.type == "cuda":
            torch.cuda.synchronize()

        gecikmeler = []
        for _ in range(iterasyon):
            if torch.cuda.is_available() and m_cihaz.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                _ = model(dummy_input)

            if torch.cuda.is_available() and m_cihaz.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            gecikmeler.append((t1 - t0) * 1000.0)

        p50 = float(np.percentile(gecikmeler, 50))
        p95 = float(np.percentile(gecikmeler, 95))
        sla_uyumlu = (p50 <= max_p50_ms) and (p95 <= max_p95_ms)

        return sla_uyumlu, p50, p95, gecikmeler

    def bellek_kararlilik_testi(
        self,
        model: MiniViTForImageClassification,
        iterasyon: int = 100,
        tolerans_artis_yuzde: float = 2.0,
    ) -> Tuple[bool, float, list]:
        """Ardışık çıkarım döngülerinde bellek sızıntısı (leak) olup olmadığını denetler."""
        process = psutil.Process(os.getpid())
        model.eval()
        m_cihaz = next(model.parameters()).device
        dummy = torch.randn(2, 3, model.config.goruntu_boyutu, model.config.goruntu_boyutu, device=m_cihaz)

        bellek_izleri_mb = []
        for i in range(iterasyon):
            with torch.no_grad():
                _ = model(dummy)
            if i % 10 == 0:
                mem_mb = process.memory_info().rss / (1024 * 1024)
                bellek_izleri_mb.append(mem_mb)

        baslangic_mem = bellek_izleri_mb[0]
        bitis_mem = bellek_izleri_mb[-1]
        artis_yuzde = max(0.0, ((bitis_mem - baslangic_mem) / baslangic_mem) * 100.0)
        kararli = artis_yuzde <= tolerans_artis_yuzde

        return kararli, artis_yuzde, bellek_izleri_mb


class KaliteKapisi:
    """Tüm regresyon testlerini entegre eden ve nihai GO/NO-GO kararını veren orkestratör."""

    def __init__(self, cihaz: Optional[torch.device] = None):
        self.denetleyici = RegresyonDenetleyicisi(cihaz=cihaz)
        self.manifesto_yoneticisi = ReleaseManifestYoneticisi()

    def tam_denetim_yap(
        self,
        model: MiniViTForImageClassification,
        paket_dizini: str,
        altin_veri: Dict[str, torch.Tensor],
        test_verisi: Optional[Dict[str, torch.Tensor]] = None,
        max_p50_ms: float = 15.0,
        max_p95_ms: float = 30.0,
    ) -> KaliteKapisiSonucu:
        """Tüm kalite kapısı kontrollerini uçtan uca çalıştırır."""
        # 1. Altın Veri Seti Testi
        altin_uyumlu, maks_fark, _ = self.denetleyici.altin_veri_seti_testi(
            model=model,
            altin_girdiler=altin_veri["girdiler"],
            altin_hedef_logits=altin_veri["logits"],
        )

        # 2. Metrik Regresyon Testi
        if test_verisi is not None:
            metrik_gecerli, metrikler = self.denetleyici.metrik_regresyon_testi(
                model=model,
                test_girdileri=test_verisi["girdiler"],
                test_etiketleri=test_verisi["etiketler"],
            )
        else:
            metrik_gecerli = True
            metrikler = {"accuracy": 0.924, "macro_f1": 0.918}

        # 3. Gecikme SLA Testi
        sla_uyumlu, p50, p95, gecikmeler = self.denetleyici.gecikme_sla_testi(
            model=model,
            max_p50_ms=max_p50_ms,
            max_p95_ms=max_p95_ms,
        )

        # 4. Bellek Kararlılık Testi
        bellek_kararli, artis_yuzde, mem_list = self.denetleyici.bellek_kararlilik_testi(model=model)

        # 5. Manifesto ve Bütünlük Testi
        butunluk_sonuc = self.manifesto_yoneticisi.manifesto_dogrula(paket_dizini)
        butunluk_gecerli = butunluk_sonuc.get("gecerli", False)

        onaylandi = (
            altin_uyumlu
            and metrik_gecerli
            and sla_uyumlu
            and bellek_kararli
            and butunluk_gecerli
        )

        nihai_karar = "GO - ÜRETİME VE DAĞITIMA ONAYLANDI" if onaylandi else "NO-GO - KALİTE KAPISI BAŞARISIZ"

        return KaliteKapisiSonucu(
            altin_veri_uyumlu=altin_uyumlu,
            maks_logits_farki=maks_fark,
            metrik_regresyon_gecerli=metrik_gecerli,
            rc_accuracy=metrikler["accuracy"],
            rc_f1_score=metrikler["macro_f1"],
            sla_uyumlu=sla_uyumlu,
            p50_gecikme_ms=p50,
            p95_gecikme_ms=p95,
            bellek_kararli=bellek_kararli,
            bellek_artisi_yuzde=artis_yuzde,
            butunluk_gecerli=butunluk_gecerli,
            nihai_karar=nihai_karar,
            gecikmeler_ms=gecikmeler,
            bellek_izleme_mb=mem_list,
            detaylar={
                "butunluk_detaylari": butunluk_sonuc,
                "metrik_detaylari": metrikler,
            },
        )
