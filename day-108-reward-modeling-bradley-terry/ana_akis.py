"""
Day 108: Bradley-Terry Tercih Modellemesi & Skaler Ödül Fonksiyonu Eğitimi Ana Akışı.
Çiftli karşılaştırma (y_w > y_l), marjin ayrışması ve 6 panelli teşhis panosu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.odul_modeli import OdulModeli
from src.odul_laboratuvari import OdulLaboratuvari
from src.gorsellestirici import OdulGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 108: Bradley-Terry Reward Modeling & Pairwise Preference Alignment")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Tercih Veri Seti ve Model Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] Çiftli Tercih Veri Seti ve Transformer Ödül Modeli Başlatılıyor...")
    lab = OdulLaboratuvari(vocab_size=1000, dim=256, cihaz=cihaz)
    chosen, rejected = lab.sentetik_tercih_verisi_uret(cift_sayisi=350, seq_len=64)

    print(f"  * Üretilen Tercih Çifti Sayısı : {chosen.shape[0]}")
    print(f"  * Dizi Uzunluğu (Prompt+Yanıt) : {chosen.shape[1]} Token")

    model = OdulModeli(
        vocab_size=1000,
        dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=128,
        pad_token_id=0,
    )
    param_sayisi = sum(p.numel() for p in model.parameters())
    print(f"  * Ödül Modeli Parametre Sayısı : {param_sayisi:,}")

    # -------------------------------------------------------------
    # ADIM 2: Bradley-Terry Kaybı ile Ödül Modeli Eğitimi
    # -------------------------------------------------------------
    print("\n[2/3] Bradley-Terry Kaybı ile Ödül Modeli Eğitiliyor (20 Epok, Marjin=0.5)...")
    egitim_raporu = lab.odul_modeli_egit(
        model=model,
        chosen=chosen,
        rejected=rejected,
        epok_sayisi=20,
        batch_size=32,
        lr=1e-3,
        margin=0.5,
    )

    print("\n--- BRADLEY-TERRY ÖDÜL MODELİ EĞİTİM GELİŞİMİ ---")
    print(f"{'EPOK':<8} | {'BT LOSS':<12} | {'DOĞRULUK (%)':<16} | {'ORT r_w':<12} | {'ORT r_l':<12} | {'MARJİN (Δr)':<14}")
    print("-" * 95)
    for ep in range(0, len(egitim_raporu["kayiplar"]), 4):
        print(
            f"{ep+1:<8} | "
            f"{egitim_raporu['kayiplar'][ep]:>10.4f} | "
            f"%{egitim_raporu['dogruluklar'][ep]:>13.2f} | "
            f"{egitim_raporu['r_w_ort'][ep]:>10.3f} | "
            f"{egitim_raporu['r_l_ort'][ep]:>10.3f} | "
            f"{egitim_raporu['marjinler'][ep]:>12.3f}"
        )
    # Son epok
    son_ep = len(egitim_raporu["kayiplar"]) - 1
    print(
        f"{son_ep+1:<8} | "
        f"{egitim_raporu['kayiplar'][son_ep]:>10.4f} | "
        f"%{egitim_raporu['dogruluklar'][son_ep]:>13.2f} | "
        f"{egitim_raporu['r_w_ort'][son_ep]:>10.3f} | "
        f"{egitim_raporu['r_l_ort'][son_ep]:>10.3f} | "
        f"{egitim_raporu['marjinler'][son_ep]:>12.3f}"
    )
    print("-" * 95)

    hacking_raporu = lab.reward_hacking_analizi(model)
    print("\n[-] REWARD HACKING & GÜVENİLİRLİK RAPORU:")
    print(f"  * Kaliteli Yanıt Ödülü (Normal) : {hacking_raporu['odul_kaliteli']:+.3f}")
    print(f"  * Uzun Tekrarlı Yanıt Ödülü     : {hacking_raporu['odul_uzun_tekrar']:+.3f}")
    print(f"  * Düşük Kaliteli Yanıt Ödülü    : {hacking_raporu['odul_kotu']:+.3f}")
    print(f"  * Net Ayrışma Güveni (Δr)       : {hacking_raporu['ayrisma_guvenilirligi']:+.3f} Puan")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Bradley-Terry Teşhis Panosu Çiziliyor...")
    gorsellestirici = OdulGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "bradley_terry_reward_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        hacking_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 108: Bradley-Terry Reward Modeling Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
