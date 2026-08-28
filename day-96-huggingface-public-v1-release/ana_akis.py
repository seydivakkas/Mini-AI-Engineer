"""
Day 96: MiniViT v1.0 Hugging Face Canlı Dağıtım ve Spaces Demo Ana Akışı.
Model paketleme, Hugging Face Hub hazırlığı, pipeline testi, Gradio demo simülasyonu
ve 6-panelli canlı dağıtım teşhis panosu üretimini yönetir.
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.dagitim_yoneticisi import HfDagitimYoneticisi, MiniViTPipeline
from src.canli_demo import GradioDemoOlusturucu
from src.gorsellestirici import PublicReleaseGorsellestirici


def main():
    print("=" * 85)
    print(">>> Day 96: MiniViT v1.0 Hugging Face Canli Dagitim ve Spaces Web Demosu")
    print("=" * 85)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Calisma Ortami Cihazi: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Model Mimarisi ve Konfigürasyon
    # -------------------------------------------------------------
    print("\n[1/5] MiniViT v1.0 Mimarisi Hazirlaniyor...")
    torch.manual_seed(42)

    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        kanal_sayisi=3,
        gizli_boyut=128,
        katman_sayisi=4,
        dikkat_baslik_sayisi=4,
        ileri_besleme_boyutu=256,
        dropout=0.1,
        sinif_sayisi=10,
    )
    model = MiniViTForImageClassification(config).to(cihaz)
    model.eval()

    toplam_param = sum(p.numel() for p in model.parameters())
    print(f"  * Model Turu            : `{config.model_type}`")
    print(f"  * Toplam Parametre      : {toplam_param:,}")
    print(f"  * Gizli Boyut / Katman  : {config.gizli_boyut}D / {config.katman_sayisi} Layers")

    # -------------------------------------------------------------
    # ADIM 2: Hugging Face Model Hub Dizinine SafeTensors Paketleme
    # -------------------------------------------------------------
    print("\n[2/5] Hugging Face Hub Formatinda Dagitim Paketi Olusturuluyor...")
    hedef_dizin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_paketi")
    dagitimci = HfDagitimYoneticisi()

    dagitim_bilgisi = dagitimci.modeli_hazirla_ve_kaydet(
        model=model,
        kayit_dizini=hedef_dizin,
        repo_id="seydivakkas/minivit-cifar10-v1",
    )

    print("=" * 85)
    print("[-] HUGGING FACE MODEL HUB DAGITIM PAKETI ICERIGI")
    print("=" * 85)
    for d_adi, boyut in dagitim_bilgisi["dosyalar"].items():
        print(f"  * {d_adi:<26}: {boyut:>8.2f} KB")
    print(f"\n  * Toplam Paket Boyutu   : {dagitim_bilgisi['toplam_boyut_kb']:.2f} KB")

    # -------------------------------------------------------------
    # ADIM 3: AutoModel ve MiniViTPipeline ile Canlı Çıkarım Testi
    # -------------------------------------------------------------
    print("\n[3/5] AutoModel ile Pipeline Yukleniyor ve Cikarim Testi Yapiliyor...")
    pipe = dagitimci.yukle_ve_pipeline_kur(hedef_dizin)

    # Örnek Test Görseli (Sentetik Uçak / Kuş Benzeri)
    ornek_resim = Image.new("RGB", (32, 32), color=(50, 120, 200))

    # Isınma
    for _ in range(3):
        _ = pipe(ornek_resim, top_k=5)

    tahminler = pipe(ornek_resim, top_k=5)

    print("=" * 85)
    print("[-] CANLI GORUNTU SINIFLANDIRMA TAHMINLERI (TOP-5)")
    print("=" * 85)
    for i, t in enumerate(tahminler, 1):
        print(f"  {i}. {t['label']:<12}: %{t['score']*100:>6.2f} Olasilik")

    # -------------------------------------------------------------
    # ADIM 4: Gecikme Benchmark'ı (Ham PyTorch vs Pipeline)
    # -------------------------------------------------------------
    print("\n[4/5] Gecikme Benchmark'i Olculuyor...")
    dummy_tensor = torch.randn(1, 3, 32, 32, device=cihaz)

    # Ham PyTorch Çıkarım
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_tensor)

    ham_gecikmeler = []
    for _ in range(30):
        if torch.cuda.is_available() and cihaz.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_tensor)
        if torch.cuda.is_available() and cihaz.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ham_gecikmeler.append((t1 - t0) * 1000.0)

    pipe_gecikmeler = []
    for _ in range(30):
        t_res = pipe(ornek_resim, top_k=5)
        pipe_gecikmeler.append(t_res[0]["gecikme_ms"])

    ham_p50 = float(np.percentile(ham_gecikmeler, 50))
    pipeline_p50 = float(np.percentile(pipe_gecikmeler, 50))

    gecikme_istatistikleri = {
        "ham_gecikme_ms": ham_p50,
        "pipeline_gecikme_ms": pipeline_p50,
    }
    print(f"  * Ham PyTorch Gecikmesi (P50)   : {ham_p50:.2f} ms")
    print(f"  * MiniViTPipeline Gecikmesi     : {pipeline_p50:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 5: Gradio Demo Testi ve 6-Panelli Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[5/5] Gradio Demo Motoru Test Ediliyor ve Teshis Panosu Ciziliyor...")
    demo_olusturucu = GradioDemoOlusturucu(pipe)
    demo_app = demo_olusturucu.arayuz_olustur()
    print("  * Gradio Blocks Demo Arayuzu Basariyla Olusturuldu.")

    pano_yolu = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar", "huggingface_public_release_paneli.png")
    gorsellestirici = PublicReleaseGorsellestirici(dpi=300)
    gorsellestirici.pano_olustur(
        dagitim_bilgisi=dagitim_bilgisi,
        ornek_tahminler=tahminler,
        gecikme_istatistikleri=gecikme_istatistikleri,
        kayit_yolu=pano_yolu,
    )

    print("\n[OK] Day 96: MiniViT v1.0 Hugging Face Canli Dagitimi ve Spaces Demosu Basariyla Tamamlandi!")


if __name__ == "__main__":
    main()
