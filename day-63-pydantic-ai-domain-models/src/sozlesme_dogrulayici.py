"""
Pydantic v2 Doğrulama Motoru, JSON Şema Çıkarıcı ve Yüksek Hızlı Benchmark Testi.
"""

from typing import Dict, Any, Tuple, Optional, List
import time
import json
from pydantic import ValidationError
import numpy as np

from .domain_modelleri import (
    GorselMetadatasi,
    BoundingBoxModeli,
    NesneTespitiSonucu,
    VektorEmbeddingSozlesmesi,
    InferenceIstekSozlesmesi,
    InferenceYanitSozlesmesi
)


class SozlesmeDogrulayici:
    """Girdi ve çıktı payload'larını Pydantic v2 ile doğrulayan ve hata raporlayan servis."""

    @staticmethod
    def dogrula_istek(payload: Dict[str, Any]) -> Tuple[Optional[InferenceIstekSozlesmesi], Optional[str]]:
        """Gelen API isteğini doğrular."""
        try:
            istek = InferenceIstekSozlesmesi.model_validate(payload)
            return istek, None
        except ValidationError as e:
            return None, e.json(indent=2)

    @staticmethod
    def dogrula_yanit(payload: Dict[str, Any]) -> Tuple[Optional[InferenceYanitSozlesmesi], Optional[str]]:
        """Model yanıtını doğrular."""
        try:
            yanit = InferenceYanitSozlesmesi.model_validate(payload)
            return yanit, None
        except ValidationError as e:
            return None, e.json(indent=2)

    @staticmethod
    def json_sema_uret(model_sinifi: Any = InferenceIstekSozlesmesi) -> Dict[str, Any]:
        """LLM Structured Outputs ve OpenAPI için JSON Schema üretir."""
        return model_sinifi.model_json_schema()


class PydanticBenchmarkEngine:
    """Pydantic v2 Rust çekirdeğinin (pydantic-core) doğrulama ve serileştirme performansını ölçer."""

    @staticmethod
    def sentetik_veri_olustur(num_samples: int = 5_000) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Geçerli ve geçersiz sentetik API istek payload'ları üretir."""
        gecerli_list = []
        gecersiz_list = []

        for i in range(num_samples):
            gecerli = {
                "istek_id": f"req_trace_{i:06d}",
                "model_adi": "YOLOv8-Vision-V2",
                "gorsel_meta": {
                    "genislik": 1920,
                    "yukseklik": 1080,
                    "kanal_sayisi": 3,
                    "format": "JPEG",
                    "dosya_boyutu_kb": 1450.5
                },
                "nms_esigi": 0.45,
                "guven_esigi": 0.50
            }
            gecerli_list.append(gecerli)

            # Geçersiz payload (ör: hatalı format, negatif boyut, ters koordinat)
            gecersiz = {
                "istek_id": f"short_{i}",  # min_length=8 ihlali
                "model_adi": "M",
                "gorsel_meta": {
                    "genislik": -100,      # gt=0 ihlali
                    "yukseklik": 1080,
                    "kanal_sayisi": 5,     # le=4 ihlali
                    "format": "GIF",       # regex format ihlali
                    "dosya_boyutu_kb": 999999.0
                },
                "nms_esigi": 1.5           # le=1.0 ihlali
            }
            gecersiz_list.append(gecersiz)

        return gecerli_list, gecersiz_list

    @classmethod
    def calistir_benchmark(cls, num_samples: int = 5_000) -> Dict[str, Any]:
        """Doğrulama hızı, serileştirme süresi ve hata yakalama oranını kıyaslar."""
        gecerli_payloads, gecersiz_payloads = cls.sentetik_veri_olustur(num_samples=num_samples)

        # 1. Geçerli Veri Doğrulama (Validation Throughput)
        start = time.perf_counter()
        dogrulanan_nesneler = []
        for p in gecerli_payloads:
            obj = InferenceIstekSozlesmesi.model_validate(p)
            dogrulanan_nesneler.append(obj)
        val_sure_s = time.perf_counter() - start
        val_qps = float(num_samples / max(val_sure_s, 1e-6))
        val_lat_us = float((val_sure_s / num_samples) * 1_000_000.0)

        # 2. Serileştirme Hızı (model_dump_json - Rust core)
        start = time.perf_counter()
        for obj in dogrulanan_nesneler:
            _ = obj.model_dump_json()
        ser_sure_s = time.perf_counter() - start
        ser_qps = float(num_samples / max(ser_sure_s, 1e-6))

        # 3. Geçersiz Veri Hata Yakalama (Error Interception)
        yakalanan_hata_sayisi = 0
        start = time.perf_counter()
        for p in gecersiz_payloads:
            try:
                _ = InferenceIstekSozlesmesi.model_validate(p)
            except ValidationError:
                yakalanan_hata_sayisi += 1
        err_sure_s = time.perf_counter() - start
        err_oran = float((yakalanan_hata_sayisi / num_samples) * 100.0)

        return {
            "toplam_ornek_sayisi": num_samples,
            "dogrulama_qps": val_qps,
            "dogrulama_gecikme_mikrosaniye": val_lat_us,
            "serilestirme_qps": ser_qps,
            "hata_yakalama_orani_yuzde": err_oran,
            "toplam_dogrulama_suresi_s": val_sure_s,
            "toplam_serilestirme_suresi_s": ser_sure_s
        }
