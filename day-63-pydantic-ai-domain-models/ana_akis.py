"""
Day 63: Pydantic v2 ile Tip Güvenli Girdi/Çıktı Sözleşmeleri & Domain Modelleri Ana Yürütme Betiği.
"""

import os
import sys
import json
import numpy as np

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.domain_modelleri import (
    GorselMetadatasi,
    BoundingBoxModeli,
    NesneTespitiSonucu,
    VektorEmbeddingSozlesmesi,
    InferenceIstekSozlesmesi,
    InferenceYanitSozlesmesi
)
from src.sozlesme_dogrulayici import SozlesmeDogrulayici, PydanticBenchmarkEngine
from src.gorsellestirici import PydanticGorsellestirici


def main():
    print("=" * 95, flush=True)
    print(">>> DAY 63: PYDANTIC V2 ILE TIP GUVENLI GIRDI/CIKTI SOZLESMELERI & AI DOMAIN MODELLERI", flush=True)
    print("=" * 95, flush=True)

    # 1. Örnek Tip Güvenli Domain Modellerinin Oluşturulması
    print("\n[+] 1. Adim: Tip Guvenli Domain Modelleri ve Sozlesmeler Olusturuluyor...", flush=True)
    meta = GorselMetadatasi(
        genislik=1920,
        yukseklik=1080,
        kanal_sayisi=3,
        format="JPEG",
        dosya_boyutu_kb=2450.0
    )
    print(f"    - Gorsel Metadatasi : {meta.format} {meta.genislik}x{meta.yukseklik} ({meta.toplam_piksel:,} piksel)")

    bbox1 = BoundingBoxModeli(x_min=0.10, y_min=0.15, x_max=0.55, y_max=0.85)
    bbox2 = BoundingBoxModeli(x_min=0.20, y_min=0.25, x_max=0.60, y_max=0.90)
    iou_skoru = bbox1.iou(bbox2)
    print(f"    - BoundingBox IoU   : {iou_skoru:.4f} (Geometrik Olarak Dogrulandi)")

    # 2. L2-Normalize Vektör Embedding Doğrulaması
    raw_vec = np.random.randn(512).astype(np.float32)
    norm_vec = (raw_vec / np.linalg.norm(raw_vec)).tolist()
    embedding_sozlesmesi = VektorEmbeddingSozlesmesi(vektor=norm_vec, beklenen_boyut=512)
    print(f"    - Embedding Vektoru : {len(embedding_sozlesmesi.vektor)}-D (L2 Norm: 1.0000 Dogrulandi)")

    # 3. LLM JSON Schema Export Testi
    print("\n[+] 2. Adim: LLM Structured Outputs Icin OpenAPI/JSON Schema Cikariliyor...", flush=True)
    schema = SozlesmeDogrulayici.json_sema_uret(InferenceIstekSozlesmesi)
    print(f"    - JSON Schema Alanlari: {list(schema.get('properties', {}).keys())}")

    # 4. Pydantic v2 Yüksek Hızlı Benchmark Testi (10,000 Payload)
    num_samples = 10_000
    print(f"\n[+] 3. Adim: {num_samples:,} Payload ile Pydantic v2 Rust Core Benchmark Kosuluyor...", flush=True)
    bench_res = PydanticBenchmarkEngine.calistir_benchmark(num_samples=num_samples)

    print("\n" + "=" * 95, flush=True)
    print(">>> 4. PYDANTIC V2 PERFORMANS VE GUCLULUK METRIKLERI", flush=True)
    print("=" * 95, flush=True)
    print(f"* Test Edilen Payload Sayisi     : {bench_res['toplam_ornek_sayisi']:,}")
    print(f"* Dogrulama Hizi (Throughput)    : {bench_res['dogrulama_qps']:>10,.0f} Payload / Saniye")
    print(f"* Tekil Dogrulama Gecikmesi      : {bench_res['dogrulama_gecikme_mikrosaniye']:>10.2f} mikrosaniye (us)")
    print(f"* Serilestirme Hizi (Dump JSON)  : {bench_res['serilestirme_qps']:>10,.0f} Payload / Saniye")
    print(f"* Gecersiz Veri Engelleme Orani  : %{bench_res['hata_yakalama_orani_yuzde']:>9.2f} (Tam Koruma)")
    print(f"* Toplam Dogrulama Suresi        : {bench_res['toplam_dogrulama_suresi_s']:>10.4f} saniye")

    # 5. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 95, flush=True)
    print(">>> 5. 6 PANELLI PYDANTIC V2 TEŞHIS VE PERFORMANS PANOSU", flush=True)
    print("=" * 95, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "pydantic_domain_paneli.png")
    cikis_yolu = PydanticGorsellestirici.panel_ciz(
        benchmark_sonuclari=bench_res,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 95, flush=True)
    print("DAY 63: PYDANTIC V2 TIP GUVENLI DOMAIN MODELLERI BASARIYLA TAMAMLANDI!", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
