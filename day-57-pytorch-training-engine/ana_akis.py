"""
Day 57: Modüler PyTorch Eğitim Motoru, Checkpoint, Early Stopping, Gradient Clipping Ana Yürütme Betiği.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Dizin yolu ekleme
MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.geri_cagirimlar import ModelCheckpointCallback, EarlyStoppingCallback, MetrikKayitCallback
from src.egitim_motoru import EgitimMotoru
from src.gorsellestirici import EgitimMotoruGorsellestirici


def veri_yukleyicileri_hazirla(num_samples: int = 1200, input_dim: int = 64, num_classes: int = 4, batch_size: int = 32):
    """Eğitim ve doğrulama için sentetik çok sınıflı veri setleri üretir."""
    np.random.seed(42)
    torch.manual_seed(42)

    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)

    ayrim = int(num_samples * 0.8)
    train_x, train_y = torch.from_numpy(X[:ayrim]), torch.from_numpy(y[:ayrim])
    val_x, val_y = torch.from_numpy(X[ayrim:]), torch.from_numpy(y[ayrim:])

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def model_olustur(input_dim: int = 64, num_classes: int = 4) -> nn.Module:
    """Modüler MLP sınıflandırma modeli."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )


def main():
    print("=" * 85, flush=True)
    print(">>> DAY 57: MODÜLER PYTORCH EĞİTİM MOTORU, CHECKPOINT & EARLY STOPPING", flush=True)
    print("=" * 85, flush=True)

    # 1. Veri Yükleyicilerin ve Modelin Hazırlanması
    print("\n[+] 1. Adım: Veri Yükleyicileri ve Model Mimarisi Başlatılıyor...", flush=True)
    train_loader, val_loader = veri_yukleyicileri_hazirla()
    model = model_olustur()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)

    # 2. Geri Çağırımların (Callbacks) Konfigürasyonu
    checkpoint_dir = os.path.join(MEVCUT_DIZIN, "checkpoints")
    checkpoint_cb = ModelCheckpointCallback(kayit_dizini=checkpoint_dir, monitor="val_loss", mode="min")
    early_stop_cb = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=4, min_delta=1e-3)
    metrik_cb = MetrikKayitCallback()

    # 3. Eğitim Motorunun Başlatılması ve Eğitimin Koşturulması
    print("\n[+] 2. Adım: Eğitim Motoru Başlatılıyor (Max Grad Norm: 1.0, Callbacks Aktif)...", flush=True)
    motor = EgitimMotoru(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        max_grad_norm=1.0,
        callbacks=[metrik_cb, checkpoint_cb, early_stop_cb]
    )

    gecmis = motor.fit(train_loader=train_loader, val_loader=val_loader, epochs=12)

    # 4. Checkpoint Resume (Geri Yükleme) Testi
    print("\n[+] 3. Adım: Kaydedilen En İyi Modelden (Resume) Geri Yükleme Test Ediliyor...", flush=True)
    en_iyi_model_yolu = os.path.join(checkpoint_dir, "en_iyi_model.pt")
    if os.path.exists(en_iyi_model_yolu):
        yuklenen = motor.resume(en_iyi_model_yolu)
        print(f"    - En İyi Model Epoch  : {yuklenen['epoch']}")
        print(f"    - En Düşük Val Loss   : {yuklenen['metrikler']['val_loss']:.4f}")
        print(f"    - Doğrulama Doğruluğu : %{yuklenen['metrikler']['val_acc']:.2f}")

    # 5. 6 Panelli Teşhis Panosunun Üretilmesi
    print("\n" + "=" * 85, flush=True)
    print(">>> 4. 6 PANELLİ EĞİTİM MOTORU PERFORMANS PANOSUNUN ÜRETİLMESİ", flush=True)
    print("=" * 85, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "egitim_motoru_paneli.png")
    cikis_yolu = EgitimMotoruGorsellestirici.panel_ciz(
        gecmis=gecmis,
        en_iyi_epoch=checkpoint_cb.en_iyi_epoch,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 85, flush=True)
    print("DAY 57: MODÜLER PYTORCH EĞİTİM MOTORU BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
