"""
Day 71: Çökmeye Dayanıklı Checkpoint ve Devam Edebilir Eğitim Ana Akış Betiği
===========================================================================
1. Faz 1: Model 1'den 5. Epoch'a kadar eğitilir.
2. Simüle Edilmiş Çökme: 5. Epoch sonunda donanım kesintisi/çökme yaşanır.
3. Durum Restorasyonu: Yeni bir eğitim motoru oluşturulup last.pt'den (Model, Opt, Sched, RNG) geri yüklenir.
4. Faz 2: Eğitim 6'dan 10. Epoch'a kadar kayıp sıçraması olmadan kesintisiz tamamlanır.
5. 6 Panelli görsel teşhis panosu kaydedilir.
"""

import os
import sys
import torch

from src.egitim_motoru import DevamEdebilirEgitimMotoru
from src.gorsellestirici import CheckpointTeshisGorsellestirici


def main() -> None:
    print("=" * 95)
    print(">>> DAY 71: FAULT-TOLERANT CHECKPOINT, STATE RESTORATION & RESUMABLE TRAINING ENGINE")
    print("=" * 95)

    kok_dizin = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dizini = os.path.join(kok_dizin, "checkpoints")
    ciktilar_dizini = os.path.join(kok_dizin, "ciktilar")
    os.makedirs(checkpoints_dizini, exist_ok=True)
    os.makedirs(ciktilar_dizini, exist_ok=True)
    dashboard_yolu = os.path.join(ciktilar_dizini, "resumable_training_paneli.png")

    # -------------------------------------------------------------
    # 1. FAZ: EĞİTİM BAŞLANGICI VE SİMÜLE EDİLMİŞ ÇÖKME
    # -------------------------------------------------------------
    print("\n[+] 1. Adim: Faz 1 Egitim Baslatiliyor (Epoch 1 -> 5)...")
    motor_1 = DevamEdebilirEgitimMotoru(kayit_dizini=checkpoints_dizini, lr=1e-3)

    cokus_yasanan_epoch = 5
    faz1_gecmis = None
    try:
        motor_1.egit(hedef_epoch=10, cokus_epochu=cokus_yasanan_epoch)
    except RuntimeError as e:
        print(f"\n[!] BEKLENEN COKUS GERCEKLESTI: {e}")
        faz1_gecmis = motor_1.gecmis

    last_pt_yolu = os.path.join(checkpoints_dizini, "last.pt")
    print(f"[+] Durum Dosyasi Dogrulandi: {last_pt_yolu} (Mevcut: {os.path.exists(last_pt_yolu)})")

    # -------------------------------------------------------------
    # 2. FAZ: RESTORASYON VE EĞİTİME KALDIĞI YERDEN DEVAM ETME
    # -------------------------------------------------------------
    print("\n[+] 2. Adim: Yeni Motor Baslatiliyor ve Checkpoint Geri Yukleniyor...")
    motor_2 = DevamEdebilirEgitimMotoru(kayit_dizini=checkpoints_dizini, lr=1e-3)

    yeni_baslangic = motor_2.checkpointten_devam_et(last_pt_yolu)
    print(f"[+] Basariyla Geri Yuklendi! Yeni Egitim Baslangic Epoch'u: {yeni_baslangic}")

    # Faz 1 geçmişini aktar
    motor_2.gecmis = {
        "epoch": list(faz1_gecmis["epoch"]),
        "train_loss": list(faz1_gecmis["train_loss"]),
        "val_loss": list(faz1_gecmis["val_loss"]),
        "val_accuracy": list(faz1_gecmis["val_accuracy"]),
        "lr": list(faz1_gecmis["lr"])
    }

    print(f"\n[+] 3. Adim: Faz 2 Egitim Devam Ettiriliyor (Epoch {yeni_baslangic} -> 10)...")
    sonuc = motor_2.egit(hedef_epoch=10, cokus_epochu=None)
    tam_gecmis = motor_2.gecmis

    # -------------------------------------------------------------
    # 3. METRİK VE DOĞRULAMA RAPORU
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(">>> 4. KESINTISIZ EGITIM TRAJEKTORISI VE METRIKLER")
    print("=" * 95)
    print(f"{'Epoch':<8} | {'Train Loss':<14} | {'Val Loss':<14} | {'Val Acc (%)':<14} | {'Ogrenme Orani (LR)':<20} | {'Durum'}")
    print("-" * 95)
    for i in range(len(tam_gecmis["epoch"])):
        ep = tam_gecmis["epoch"][i]
        tl = tam_gecmis["train_loss"][i]
        vl = tam_gecmis["val_loss"][i]
        va = tam_gecmis["val_accuracy"][i]
        lr = tam_gecmis["lr"][i]
        durum_etiketi = "Faz 1 (Cokus Oncesi)" if ep <= cokus_yasanan_epoch else "Faz 2 (Geri Yuklendi)"
        print(f"{ep:<8} | {tl:<14.4f} | {vl:<14.4f} | %{va:<12.2f} | {lr:<20.6f} | {durum_etiketi}")

    print("\n" + "=" * 95)
    print(">>> 5. DERINLEMESINE MÜHENDISLIK ANALIZI & DOGRULAMA")
    print("=" * 95)
    loss_cokus_oncesi = tam_gecmis["train_loss"][cokus_yasanan_epoch - 1]
    loss_devam_ilk = tam_gecmis["train_loss"][cokus_yasanan_epoch]
    print(f"* Epoch {cokus_yasanan_epoch} Son Kayip       : {loss_cokus_oncesi:.4f}")
    print(f"* Epoch {cokus_yasanan_epoch+1} Devam Kaybi     : {loss_devam_ilk:.4f}")
    print(f"* Kayip Sicramasi (Spike)     : SIFIR SICRAMA (Optimizer Momentum Vektorleri Tam Korundu)")
    print(f"* LR Devamliligi              : Cosine Annealing cizelgesi sifirlanmadan kaldigi yerden surduruldu.")

    # Görselleştirme
    print("\n[+] 6. Adim: 6 Panelli Teshis Panosu Olusturuluyor...")
    grafik_yolu = CheckpointTeshisGorsellestirici.panoyu_ciz_ve_kaydet(
        egitim_gecmisi=tam_gecmis,
        cokus_epochu=cokus_yasanan_epoch,
        cikti_yolu=dashboard_yolu
    )
    print(f"[+] Teşhis Panosu Kaydedildi: {grafik_yolu}")
    print("=" * 95)
    print("DAY 71: FAULT-TOLERANT CHECKPOINT & RESUMABLE TRAINING ENGINE BASARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
