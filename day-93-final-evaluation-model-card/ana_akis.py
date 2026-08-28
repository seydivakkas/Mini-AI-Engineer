"""
Day 93: Kapsamlı Değerlendirme, Yanlılık Testleri ve Model Card Laboratuvarı
---------------------------------------------------------------------------
Model Performansı, Dilim (Slice) Analizi, Adillik (Fairness) ve MODEL_CARD.md Üretimi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model import FinalVisionClassifier
from src.metrik_hesaplayici import MetrikHesaplayici
from src.yanlilik_denetleyicisi import YanlilikDenetleyicisi
from src.model_card_uretici import ModelCardUretici, ModelMetadata
from src.gorsellestirici import DegerlendirmeGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 85)
    print("🚀 Day 93: Kapsamlı Değerlendirme, Yanlılık (Bias) Testleri ve Model Card Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Model ve Değerlendirme Bileşenlerinin Hazırlığı
    sinif_sayisi = 10
    model = FinalVisionClassifier(giris_kanali=3, sinif_sayisi=sinif_sayisi, taban_filtre=32).to(cihaz)
    hesaplayici = MetrikHesaplayici(sinif_sayisi=sinif_sayisi, ece_kutu_sayisi=10)
    yanlilik_denetleyicisi = YanlilikDenetleyicisi(adillik_esigi=0.80, maks_fark_esigi=0.15)
    card_uretici = ModelCardUretici(
        metadata=ModelMetadata(
            model_adi="MiniVision-CIFAR10-v1",
            surum="v1.0.0-rc",
            yazar="Seydi Eryılmaz (@seydivakkas)",
            parametre_sayisi=sum(p.numel() for p in model.parameters()),
            test_veri_seti_boyutu=500,
        )
    )

    # -------------------------------------------------------------
    # ADIM 1: Sentetik Çoklu Dilim (Multi-Slice) Test Seti Üretimi
    # -------------------------------------------------------------
    print("\n[1/4] Çoklu Dilim (Multi-Slice) Test Veri Seti Üretiliyor...")
    toplam_ornek = 500

    # 4 Farklı Dilim Oluştur
    # 1. Standart (150)
    # 2. Düşük Işık (120)
    # 3. Yüksek Kontrast (120)
    # 4. Gürültülü / Bozuk (110)
    x_standart = torch.randn(150, 3, 32, 32)
    x_karanlik = torch.randn(120, 3, 32, 32) * 0.4 - 0.8
    x_parlak = torch.randn(120, 3, 32, 32) * 1.5 + 1.0
    x_gurultu = torch.randn(110, 3, 32, 32) + torch.randn(110, 3, 32, 32) * 0.8

    tum_x = torch.cat([x_standart, x_karanlik, x_parlak, x_gurultu], dim=0)
    tum_y = torch.randint(0, sinif_sayisi, (toplam_ornek,))

    dilim_maskeleri = {
        "Standart (Temiz)": np.zeros(toplam_ornek, dtype=bool),
        "Düşük Işık (Karanlık)": np.zeros(toplam_ornek, dtype=bool),
        "Yüksek Kontrast": np.zeros(toplam_ornek, dtype=bool),
        "Gürültülü / Bozuk": np.zeros(toplam_ornek, dtype=bool),
    }
    dilim_maskeleri["Standart (Temiz)"][:150] = True
    dilim_maskeleri["Düşük Işık (Karanlık)"][150:270] = True
    dilim_maskeleri["Yüksek Kontrast"][270:390] = True
    dilim_maskeleri["Gürültülü / Bozuk"][390:] = True

    print(f"  ✓ Toplam {toplam_ornek} Örnek ve 4 Bağımsız Dilim Başarıyla Hazırlandı.")

    # -------------------------------------------------------------
    # ADIM 2: Model Çıkarımı ve Genel Performans Metrikleri
    # -------------------------------------------------------------
    print("\n[2/4] Model Çıkarımı ve Kapsamlı Metrikler Hesaplanıyor...")
    tum_x_dev = tum_x.to(cihaz)
    with torch.no_grad():
        logitler = model(tum_x_dev)
        olasiliklar_t = torch.softmax(logitler, dim=-1)
        tahminler_t = torch.argmax(olasiliklar_t, dim=-1)

    y_gercek_np = tum_y.numpy()
    y_tahmin_np = tahminler_t.cpu().numpy()
    olasiliklar_np = olasiliklar_t.cpu().numpy()

    metrikler = hesaplayici.hesapla(y_gercek_np, y_tahmin_np, olasiliklar=olasiliklar_np)

    print("=" * 85)
    print("📊 MODEL NİCEL DEĞERLENDİRME SONUÇLARI")
    print("=" * 85)
    print(f"• Toplam Test Örneği   : {metrikler.toplam_ornek}")
    print(f"• Genel Doğruluk (Acc) : %{metrikler.dogruluk * 100:.2f}")
    print(f"• Macro F1-Skoru       : {metrikler.macro_f1:.4f}")
    print(f"• Weighted F1-Skoru    : {metrikler.weighted_f1:.4f}")
    print(f"• Macro Precision      : {metrikler.macro_precision:.4f}")
    print(f"• Macro Recall         : {metrikler.macro_recall:.4f}")
    print(f"• ECE Kalibrasyon      : {metrikler.kalibrasyon.ece_skoru:.4f}")
    print(f"• Brier Skoru          : {metrikler.kalibrasyon.brier_skoru:.4f}")

    # -------------------------------------------------------------
    # ADIM 3: Dilim (Slice) ve Adillik / Yanlılık Denetimi
    # -------------------------------------------------------------
    print("\n[3/4] Alt Grup Dilimleri (Fairness / Bias) Denetleniyor...")
    adillik = yanlilik_denetleyicisi.dilimleri_degerlendir(
        y_gercek=y_gercek_np,
        y_tahmin=y_tahmin_np,
        dilim_maskeleri=dilim_maskeleri,
    )

    print("─" * 85)
    print("⚖️ ALT GRUP DİLİM VE ADİLLİK (FAIRNESS) ANALİZİ")
    print("─" * 85)
    for d_adi, d_sonuc in adillik.dilim_sonuclari.items():
        print(f"  • {d_adi:<24}: Örnek: {d_sonuc.ornek_sayisi:>3} | Doğruluk: %{d_sonuc.dogruluk * 100:>5.2f} | F1: {d_sonuc.f1_skoru:.4f}")

    print(f"\n• Disparate Impact Oranı : %{adillik.disparate_impact_orani * 100:.2f} (Eşik: >= %80.0)")
    print(f"• Maksimum Dilim Farkı   : %{adillik.maks_dogruluk_farki * 100:.2f}")
    print(f"• Adillik Kararı         : {'✅ GEÇTİ' if adillik.adillik_esigi_gecti_mi else '⚠️ UYARI'}")

    # -------------------------------------------------------------
    # ADIM 4: MODEL_CARD.md ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n[4/4] Standart MODEL_CARD.md ve Teşhis Panosu Oluşturuluyor...")
    card_yolu = os.path.join(os.path.dirname(__file__), "MODEL_CARD.md")
    card_uretici.model_card_olustur(metrikler=metrikler, adillik_raporu=adillik, kayit_yolu=card_yolu)
    print(f"  ✓ Standart Model Card Oluşturuldu: {card_yolu}")

    gorsellestirici = DegerlendirmeGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "degerlendirme_ve_model_card_paneli.png")

    gorsellestirici.olustur_degerlendirme_paneli(
        metrikler=metrikler,
        adillik_raporu=adillik,
        metadata=card_uretici.metadata,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ 6-Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 93: Kapsamlı Değerlendirme ve Model Card Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
