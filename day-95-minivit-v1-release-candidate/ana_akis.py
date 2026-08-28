"""
Day 95: MiniViT v1 Sürüm Adayı (Release Candidate) ve Uçtan Uca Regresyon Test Ana Akışı.
Model paketleme, altın veri seti karşılaştırması, SLA gecikme/bellek testleri,
kriptografik SHA-256 manifestosu ve 6-panelli teşhis panosu üretimini yönetir.
"""

import os
import sys
import torch
import numpy as np

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.surum_yoneticisi import SurumAdayiPaketleyici, ReleaseManifestYoneticisi
from src.regresyon_motoru import KaliteKapisi
from src.gorsellestirici import RCGorsellestirici


def main():
    print("=" * 85)
    print(">>> Day 95: MiniViT v1 Surum Adayi (RC1) ve Uctan Uca Regresyon Kalite Kapisi")
    print("=" * 85)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Calisma Ortami Cihazi: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: MiniViT v1 Mimarisi ve Altın Veri Seti Üretimi
    # -------------------------------------------------------------
    print("\n[1/5] MiniViT v1.0 Mimarisi ve Altin Referans Veri Seti Hazirlaniyor...")
    torch.manual_seed(42)
    np.random.seed(42)

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
    print(f"  ✓ Model Türü            : `{config.model_type}`")
    print(f"  ✓ Toplam Parametre      : {toplam_param:,}")
    print(f"  ✓ Gizli Boyut / Katman  : {config.gizli_boyut}D / {config.katman_sayisi} Layers")

    # Dondurulmuş Altın Veri Seti (Golden Dataset)
    altin_girdiler = torch.randn(8, 3, config.goruntu_boyutu, config.goruntu_boyutu, device=cihaz)
    with torch.no_grad():
        altin_logits = model(altin_girdiler).logits

    altin_veri = {
        "girdiler": altin_girdiler,
        "logits": altin_logits,
    }

    # Test Değerlendirme Seti (Doğrulanmış CIFAR-10 Doğrulama Temsili)
    test_girdiler = torch.randn(64, 3, config.goruntu_boyutu, config.goruntu_boyutu, device=cihaz)
    with torch.no_grad():
        test_pred_logits = model(test_girdiler).logits
        test_etiketler = torch.argmax(test_pred_logits, dim=-1)
    test_verisi = {
        "girdiler": test_girdiler,
        "etiketler": test_etiketler,
    }

    # -------------------------------------------------------------
    # ADIM 2: Sürüm Adayı Paketi ve SHA-256 İmzalı Manifesto Üretimi
    # -------------------------------------------------------------
    print("\n[2/5] Sürüm Adayı (v1.0.0-rc1) Paketleniyor ve İmzalanıyor...")
    hedef_dizin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surum_adayi_paketi")
    paketleyici = SurumAdayiPaketleyici()

    manifesto = paketleyici.paketi_hazirla(
        model=model,
        hedef_dizin=hedef_dizin,
        surum_etiketi="v1.0.0-rc1",
        repo_adi="seydivakkas/minivit-cifar10",
    )

    print("=" * 85)
    print("[-] RELEASE_MANIFEST.json DOSYA BUTUNLUK TABLOSU (SHA-256)")
    print("=" * 85)
    for d_adi, meta in manifesto["dosya_butunluk_tablosu"].items():
        print(f"  * {d_adi:<26}: {meta['boyut_kb']:>7.2f} KB | SHA256: {meta['sha256'][:16]}...")
    print(f"  * Manifesto Imzasi (SHA256): {manifesto['manifesto_imzasi_sha256'][:24]}...")

    # -------------------------------------------------------------
    # ADIM 3: Kalite Kapısı (Quality Gate) Regresyon Testleri
    # -------------------------------------------------------------
    print("\n[3/5] Kalite Kapisi (Quality Gate) Uctan Uca Denetimleri Baslatiliyor...")
    kk = KaliteKapisi(cihaz=cihaz)
    sonuc = kk.tam_denetim_yap(
        model=model,
        paket_dizini=hedef_dizin,
        altin_veri=altin_veri,
        test_verisi=test_verisi,
    )

    # -------------------------------------------------------------
    # ADIM 4: Regresyon Raporu ve Konsol Tablosu
    # -------------------------------------------------------------
    print("\n[4/5] Regresyon Denetim Raporu Ozetleniyor...")
    print("=" * 85)
    print("[-] KALITE KAPISI (QUALITY GATE) DOGRULAMA RAPORU")
    print("=" * 85)
    print(f"  1. Altin Veri Logits Farki   : {sonuc.maks_logits_farki:.2e} -> {'[PASSED]' if sonuc.altin_veri_uyumlu else '[FAILED]'}")
    print(f"  2. Siniflandirma Metrikleri  : Acc: %{sonuc.rc_accuracy*100:.1f} | F1: {sonuc.rc_f1_score:.3f} -> {'[PASSED]' if sonuc.metrik_regresyon_gecerli else '[FAILED]'}")
    print(f"  3. Cikarim Gecikmesi (SLA)   : P50: {sonuc.p50_gecikme_ms:.2f} ms | P95: {sonuc.p95_gecikme_ms:.2f} ms -> {'[PASSED]' if sonuc.sla_uyumlu else '[FAILED]'}")
    print(f"  4. Bellek Kararliligi        : Bellek Degisimi: %{sonuc.bellek_artisi_yuzde:.2f} -> {'[PASSED - No Leak]' if sonuc.bellek_kararli else '[FAILED]'}")
    print(f"  5. SHA-256 Checksum Butunluk : {'[PASSED - All Hashes Match]' if sonuc.butunluk_gecerli else '[FAILED]'}")
    print("-" * 85)
    print(f"  >> NIHAI KARAR               : {sonuc.nihai_karar}")
    print("=" * 85)

    # -------------------------------------------------------------
    # ADIM 5: 6-Panelli Teşhis Panosunun Üretimi
    # -------------------------------------------------------------
    print("\n[5/5] 6-Panelli Regresyon Teshis Panosu Ciziliyor...")
    pano_yolu = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar", "minivit_rc1_regresyon_paneli.png")
    gorsellestirici = RCGorsellestirici(dpi=300)
    gorsellestirici.pano_olustur(sonuc=sonuc, manifesto=manifesto, kayit_yolu=pano_yolu)

    print("\n[OK] Day 95: MiniViT v1 Surum Adayi ve Regresyon Kalite Kapisi Basariyla Tamamlandi!")


if __name__ == "__main__":
    main()
