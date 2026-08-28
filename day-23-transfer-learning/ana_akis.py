"""Day 23 Ana Çalıştırma Akışı: Transfer Öğrenme ve İnce Ayar (Transfer Learning & Fine-Tuning).

Bu betik; ResNet18 ve EfficientNet-B0 mimarilerini kullanarak Sıfırdan Eğitim (Scratch),
Öznitelik Çıkarma (Feature Extraction) ve İnce Ayar (Fine-Tuning) yaklaşımlarını
deneysel olarak karşılaştırır ve sonuç raporunu üretir.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
import torch

from src.model_secici import TransferModelSecici
from src.veri_hazirlayici import TransferVeriYoneticisi
from src.egitici import TransferEgitici, TransferEgitimSonucu
from src.karsilastirici import TransferKarsilastirici
from src.gorsellestirici import TransferGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    """Day 23 transfer öğrenme tam karşılaştırma akışını yürütür."""
    baslik("AŞAMA 1: ImageNet Uyumlu Görsel Veri Setinin Hazırlanması")
    yonetici = TransferVeriYoneticisi(hedef_boyut=(64, 64), random_state=42)
    X, y, sinif_isimleri = yonetici.sentetik_veri_seti_uret(sinif_basina_ornek=40)

    print(f"[+] Toplam Görsel Sayısı: {len(X)}")
    print(f"[+] Boyutlar: {X.shape} (N={X.shape[0]}, H={X.shape[1]}, W={X.shape[2]}, C={X.shape[3]})")
    print(f"[+] Sınıflar: {sinif_isimleri}")

    train_loader, val_loader, test_loader, X_test, y_test = yonetici.veri_bol_ve_yukleyicileri_olustur(
        X, y, val_orani=0.15, test_orani=0.20, batch_size=16
    )

    print(f"[+] Eğitim Kümesi (Train DataLoader)       : {len(train_loader.dataset)} örnek ({len(train_loader)} batch)")
    print(f"[+] Doğrulama Kümesi (Validation DataLoader): {len(val_loader.dataset)} örnek ({len(val_loader)} batch)")
    print(f"[+] Test Kümesi (Test DataLoader)           : {len(test_loader.dataset)} örnek ({len(test_loader)} batch)")

    baslik("AŞAMA 2: Model Mimarilerinin ve Katman Dondurma Stratejilerinin İncelenmesi")
    # ResNet18 Feature Extraction
    r18_fe = TransferModelSecici.resnet18_olustur(num_classes=4, strateji="feature_extraction")
    p_fe = TransferModelSecici.parametre_ozeti(r18_fe)
    print(f"[+] ResNet18 (Feature Extraction): Toplam={p_fe['total']:,} | Eğitilen={p_fe['trainable']:,} (Dondurulan={p_fe['frozen']:,})")

    # ResNet18 Fine-Tuning
    r18_ft = TransferModelSecici.resnet18_olustur(num_classes=4, strateji="fine_tuning")
    p_ft = TransferModelSecici.parametre_ozeti(r18_ft)
    print(f"[+] ResNet18 (Fine-Tuning)       : Toplam={p_ft['total']:,} | Eğitilen={p_ft['trainable']:,} (Dondurulan={p_ft['frozen']:,})")

    # EfficientNet-B0 Feature Extraction
    eff_fe = TransferModelSecici.efficientnet_b0_olustur(num_classes=4, strateji="feature_extraction")
    p_eff = TransferModelSecici.parametre_ozeti(eff_fe)
    print(f"[+] EfficientNet-B0 (Feature Ext): Toplam={p_eff['total']:,} | Eğitilen={p_eff['trainable']:,} (Dondurulan={p_eff['frozen']:,})")

    baslik("AŞAMA 3: Ayrıştırılmış Öğrenme Oranlarının (Discriminative LRs) Oluşturulması")
    param_gruplari = TransferModelSecici.ayrisik_parametre_gruplari(
        r18_ft, lr_omurga=1e-4, lr_baslik=1e-3
    )
    print(f"[+] Fine-Tuning Parametre Grupları Sayısı: {len(param_gruplari)}")
    for i, g in enumerate(param_gruplari):
        print(f"    - Grup #{i+1}: Parametre Tensör Sayısı={len(g['params'])} | Öğrenme Oranı={g['lr']}")

    baslik("AŞAMA 4: Karşılaştırmalı Transfer Öğrenme Deneylerinin Koşturulması")
    karsilastirici = TransferKarsilastirici()
    sonuclar: List[TransferEgitimSonucu] = karsilastirici.karsilastirmali_deney_kos(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=4,
        epochs=15,
    )

    baslik("AŞAMA 5: Karşılaştırmalı Performans ve Verimlilik Tablosu")
    print(f"{'Model & Strateji':<35} | {'Eğitilen Param':<16} | {'Test Acc':<10} | {'F1-Macro':<10} | {'Süre (sn)':<10}")
    print("-" * 90)
    for s in sonuclar:
        print(f"{s.model_adi:<35} | {s.egitilebilir_parametre:<16,} | %{s.test_dogruluk * 100:<9.1f} | {s.f1_macro:<10.4f} | {s.egitim_suresi_sn:<10.2f}")

    baslik("AŞAMA 6: Analiz ve Teşhis Raporunun Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    rapor_dosyasi = cikti_klasor / "transfer_ogrenme_raporu.png"

    TransferGorsellestirici.karsilastirma_raporu_ciz(
        sonuclar=sonuclar,
        sinif_isimleri=sinif_isimleri,
        hedef_dosya=rapor_dosyasi,
    )
    print(f"[+] Transfer öğrenme karşılaştırma ve teşhis raporu kaydedildi: {rapor_dosyasi}")


if __name__ == "__main__":
    main()
