"""
Day 82: Öğretmen-Öğrenci Modeli Bilgi Damıtma (Knowledge Distillation)
---------------------------------------------------------------------
Derin Öğretmen Modelinin Karanlık Bilgisini (Dark Knowledge), Sıcaklık (Temperature τ)
ve KL-Diverjansı ile Hafif Öğrenci Modele Aktaran Uçtan Uca Laboratuvar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.modeller import DerinKonvolusyonelOgretmen, KompaktOgrenciModeli
from src.damitici_motor import BilgiDamiticiMotor
from src.gorsellestirici import DamitmaGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def veri_olustur(ornek_sayisi: int = 800, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    """
    Sınıflar arası hiyerarşik korelasyonlar içeren (ör. 0 ile 1 benzer, 2 ile 3 benzer)
    ve gerçekçi gürültüye sahip sentetik veri kümesi oluşturur.
    """
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 0.9
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        # Ana sınıf sinyali
        h_pos = (c % 5) * 6
        w_pos = ((c // 5) % 5) * 6
        x[i, :, h_pos:h_pos+4, w_pos:w_pos+4] += 1.2
        # Kardeş sınıf (Dark Knowledge) benzerlik sinyali
        kardes_c = (c + 1) % sinif_sayisi
        k_h = (kardes_c % 5) * 6
        k_w = ((kardes_c // 5) % 5) * 6
        x[i, :, k_h:k_h+4, k_w:k_w+4] += 0.4
    return x, y


def main():
    print("=" * 85)
    print("🚀 Day 82: Öğretmen-Öğrenci Modeli Bilgi Damıtma (Knowledge Distillation)")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    sinif_sayisi = 10
    batch_size = 32
    toplam_epok = 10

    print(f"\n[1/5] Sentetik Veri Kümeleri Hazırlanıyor...")
    tr_x, tr_y = veri_olustur(640, sinif_sayisi)
    val_x, val_y = veri_olustur(160, sinif_sayisi)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    print(f"  ✓ Eğitim Örnek Sayısı: {len(tr_x)} | Doğrulama Örnek Sayısı: {len(val_x)}")

    # 2. Öğretmen Modelinin Eğitilmesi ve Dondurulması
    print(f"\n[2/5] Derin Öğretmen Modeli Başlatılıyor ve Eğitiliyor...")
    ogretmen = DerinKonvolusyonelOgretmen(sinif_sayisi=sinif_sayisi, taban_kanal=32).to(cihaz)
    ogretmen_param = sum(p.numel() for p in ogretmen.parameters())

    opt_t = torch.optim.AdamW(ogretmen.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(8):
        ogretmen.train()
        for bx, by in tr_loader:
            bx, by = bx.to(cihaz), by.to(cihaz)
            opt_t.zero_grad()
            loss = F.cross_entropy(ogretmen(bx), by)
            loss.backward()
            opt_t.step()

    ogretmen.eval()
    val_correct = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(cihaz), by.to(cihaz)
            val_correct += (ogretmen(bx).argmax(dim=-1) == by).sum().item()
    ogretmen_acc = (val_correct / len(val_x)) * 100.0
    print(f"  ✓ Öğretmen Parametre Sayısı: {ogretmen_param:,}")
    print(f"  ✓ Öğretmen Doğrulama Doğruluğu: %{ogretmen_acc:.2f}")

    # 3. Deney 1: Bağımsız Öğrenci (Pure Cross-Entropy)
    print(f"\n[3/5] Deney 1: Bağımsız Öğrenci (Pure Cross-Entropy) Eğitiliyor...")
    ogrenci_bagimsiz = KompaktOgrenciModeli(sinif_sayisi=sinif_sayisi, taban_kanal=16).to(cihaz)
    ogrenci_param = sum(p.numel() for p in ogrenci_bagimsiz.parameters())

    motor_bagimsiz = BilgiDamiticiMotor(
        ogrenci_modeli=ogrenci_bagimsiz,
        ogretmen_modeli=None,
        cihaz=cihaz,
        ogrenme_orani=2e-3
    )
    gecmis_bagimsiz = motor_bagimsiz.egit(tr_loader, val_loader, toplam_epok=12)
    print(f"  ✓ Bağımsız Öğrenci En İyi Doğruluk: %{max(gecmis_bagimsiz['dogrulama_dogruluk']):.2f}")

    # 4. Deney 2: Damıtılmış Öğrenci (Knowledge Distillation)
    print(f"\n[4/5] Deney 2: Damıtılmış Öğrenci (KD with Teacher τ=3.0, α=0.5) Eğitiliyor...")
    tohum_belirle(42)
    ogrenci_damitilmis = KompaktOgrenciModeli(sinif_sayisi=sinif_sayisi, taban_kanal=16).to(cihaz)

    motor_damitilmis = BilgiDamiticiMotor(
        ogrenci_modeli=ogrenci_damitilmis,
        ogretmen_modeli=ogretmen,
        cihaz=cihaz,
        ogrenme_orani=2e-3,
        sicaklik=3.0,
        alfa=0.5
    )
    gecmis_damitilmis = motor_damitilmis.egit(tr_loader, val_loader, toplam_epok=12)
    print(f"  ✓ Damıtılmış Öğrenci En İyi Doğruluk: %{max(gecmis_damitilmis['dogrulama_dogruluk']):.2f}")

    fark_artis = max(gecmis_damitilmis['dogrulama_dogruluk']) - max(gecmis_bagimsiz['dogrulama_dogruluk'])
    print(f"  🚀 Knowledge Distillation Doğruluk Kazanımı: +%{fark_artis:.2f}")

    # 5. Sıcaklık Analizi ve 6 Panelli Teşhis Panosu
    print(f"\n[5/5] Sıcaklık (τ) Analizi ve 6 Panelli Teşhis Panosu Kaydediliyor...")
    ornek_girdi = val_x[0:1].to(cihaz)
    with torch.no_grad():
        ornek_logitler = ogretmen(ornek_girdi).squeeze(0)

    model_params_dict = {
        f"Öğretmen Modeli\n({ogretmen_param:,} param)": ogretmen_param,
        f"Öğrenci Modeli\n({ogrenci_param:,} param)": ogrenci_param
    }

    gorsellestirici = DamitmaGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "knowledge_distillation_paneli.png")

    gorsellestirici.olustur_damitma_paneli(
        ornek_logitler=ornek_logitler,
        bagimsiz_gecmis=gecmis_bagimsiz,
        damitilmis_gecmis=gecmis_damitilmis,
        ogretmen_acc=ogretmen_acc,
        model_parametreleri=model_params_dict,
        kayit_yolu=cikti_yolu
    )

    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 82: Knowledge Distillation Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
