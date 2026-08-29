"""
Day 113: Simple Preference Optimization (SimPO) ile LLM Hizalama Ana Akışı.
Referans modelsiz (pi_ref olmadan), hedef marjinli (gamma) ve uzunluk normalizasyonlu tercih optimizasyonu.
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

from src.simpo_laboratuvari import SimPOLaboratuvari
from src.gorsellestirici import SimPOGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 113: Simple Preference Optimization (SimPO) & Target Margin Preference Alignment")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Model ve Sentetik Tercih Veri Seti Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] SimPO Tekil Modeli ve Çiftli Tercih Veri Seti Başlatılıyor...")
    lab = SimPOLaboratuvari(
        vocab_size=1000,
        dim=256,
        num_heads=4,
        num_layers=4,
        cihaz=cihaz,
    )
    c_ids, r_ids, c_mask, r_mask = lab.sentetik_tercih_verisi_uret(
        cift_sayisi=350, prompt_len=10, resp_len=14
    )

    model_p = sum(p.numel() for p in lab.model.parameters())
    print(f"  * Üretilen Tercih Çifti Sayısı  : {c_ids.shape[0]}")
    print(f"  * Dizi Uzunluğu (Prompt+Yanıt)  : {c_ids.shape[1]} Token")
    print(f"  * 1. Politika Modeli (pi_theta) : {model_p:,} Parametre (Eğitilebilir)")
    print("  * 2. Referans Modeli (pi_ref)   : GEREK YOK (0 Parametre - %50 VRAM Tasarrufu!)")
    print("  * 3. Critic & Reward Modeli     : GEREK YOK (0 Parametre)")
    print("  * 4. Hedef Ödül Marjini (gamma) : γ = 0.5 (Zorunlu Güven Aralığı)")

    # -------------------------------------------------------------
    # ADIM 2: SimPO Eğitimi (20 Epok, beta=2.0, gamma=0.5)
    # -------------------------------------------------------------
    print("\n[2/3] SimPO Tercih Eğitimi Koşturuluyor (20 Epok, β=2.0, γ=0.5)...")
    egitim_raporu = lab.simpo_egit(
        chosen_ids=c_ids,
        rejected_ids=r_ids,
        chosen_mask=c_mask,
        rejected_mask=r_mask,
        epok_sayisi=20,
        batch_size=32,
        lr=1e-3,
        beta=2.0,
        gamma=0.5,
    )

    print("\n--- SimPO HEDEF MARJİNLİ HİZALAMA EĞİTİM GELİŞİMİ ---")
    print(f"{'EPOK':<8} | {'KAYIP (LOSS)':<12} | {'CHOSEN ÖDÜL':<12} | {'REJECTED ÖDÜL':<14} | {'ÖDÜL MARJİNİ (Δr)':<18} | {'MARJİN İHLALİ':<14} | {'DOĞRULUK (%)':<12}")
    print("-" * 95)
    for ep in range(0, len(egitim_raporu["kayiplar"]), 4):
        print(
            f"{ep+1:<8} | "
            f"{egitim_raporu['kayiplar'][ep]:>10.4f} | "
            f"{egitim_raporu['chosen_odulleri'][ep]:>10.3f} | "
            f"{egitim_raporu['rejected_odulleri'][ep]:>12.3f} | "
            f"{egitim_raporu['odul_farklari'][ep]:>16.3f} | "
            f"%{egitim_raporu['marjin_ihlalleri'][ep]:>11.2f} | "
            f"%{egitim_raporu['dogruluklar'][ep]:>10.2f}"
        )
    # Son epok
    son_ep = len(egitim_raporu["kayiplar"]) - 1
    print(
        f"{son_ep+1:<8} | "
        f"{egitim_raporu['kayiplar'][son_ep]:>10.4f} | "
        f"{egitim_raporu['chosen_odulleri'][son_ep]:>10.3f} | "
        f"{egitim_raporu['rejected_odulleri'][son_ep]:>12.3f} | "
        f"{egitim_raporu['odul_farklari'][son_ep]:>16.3f} | "
        f"%{egitim_raporu['marjin_ihlalleri'][son_ep]:>11.2f} | "
        f"%{egitim_raporu['dogruluklar'][son_ep]:>10.2f}"
    )
    print("-" * 95)

    kiyas_raporu = lab.mimari_4lu_kiyas_raporu()
    print("\n[-] PPO vs DPO vs ORPO vs SimPO 4'LÜ KIYASLAMA TABLOSU:")
    for i, ynt in enumerate(kiyas_raporu["yontemler"]):
        print(f"  * {ynt:<12}: Ref Model: {kiyas_raporu['ref_model_gereksinimi'][i]:<14} | Marjin (γ): {kiyas_raporu['hedef_marjin_gamma'][i]:<16} | VRAM: {kiyas_raporu['vram_kullanimi'][i]}")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli SimPO Hizalama Teşhis Panosu Çiziliyor...")
    gorsellestirici = SimPOGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "simpo_alignment_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        kiyas_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 113: Simple Preference Optimization (SimPO) Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
