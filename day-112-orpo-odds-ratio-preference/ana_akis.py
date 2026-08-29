"""
Day 112: Odds Ratio Preference Optimization (ORPO) ile LLM Hizalama Ana Akışı.
Tek aşamalı monolitik SFT + Tercih Optimizasyonu, Log-Odds dinamiği ve 6 panelli teşhis panosu.
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

from src.orpo_laboratuvari import ORPOLaboratuvari
from src.gorsellestirici import ORPOGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 112: Odds Ratio Preference Optimization (ORPO) & Monolithic SFT + Alignment")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Model ve Sentetik Tercih Veri Seti Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] ORPO Monolitik Modeli ve Çiftli Veri Seti Başlatılıyor...")
    lab = ORPOLaboratuvari(
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
    print(f"  * 1. Monolitik Model (pi_theta) : {model_p:,} Parametre (Eğitilebilir)")
    print("  * 2. Referans Modeli (pi_ref)   : GEREK YOK (0 Parametre - %50 Ek Tasarruf!)")
    print("  * 3. Critic & Reward Modeli     : GEREK YOK (0 Parametre - Tek Model GPU'da!)")

    # -------------------------------------------------------------
    # ADIM 2: ORPO Monolitik Eğitimi (20 Epok, lambda_or=0.5)
    # -------------------------------------------------------------
    print("\n[2/3] ORPO Tek Aşamalı SFT + Alignment Eğitimi Koşturuluyor (20 Epok, λ_OR=0.5)...")
    egitim_raporu = lab.orpo_egit(
        chosen_ids=c_ids,
        rejected_ids=r_ids,
        chosen_mask=c_mask,
        rejected_mask=r_mask,
        epok_sayisi=20,
        batch_size=32,
        lr=1e-3,
        lambda_or=0.5,
    )

    print("\n--- ORPO MONOLİTİK HİZALAMA EĞİTİM GELİŞİMİ ---")
    print(f"{'EPOK':<8} | {'TOPLAM LOSS':<12} | {'SFT KAYBI':<10} | {'OR KAYBI':<10} | {'LOG-ODDS ORANI':<16} | {'DOĞRULUK (%)':<14}")
    print("-" * 95)
    for ep in range(0, len(egitim_raporu["toplam_kayiplar"]), 4):
        print(
            f"{ep+1:<8} | "
            f"{egitim_raporu['toplam_kayiplar'][ep]:>10.4f} | "
            f"{egitim_raporu['kayiplar_sft'][ep]:>8.4f} | "
            f"{egitim_raporu['kayiplar_or'][ep]:>8.4f} | "
            f"{egitim_raporu['log_odds_oranlari'][ep]:>14.3f} | "
            f"%{egitim_raporu['dogruluklar'][ep]:>11.2f}"
        )
    # Son epok
    son_ep = len(egitim_raporu["toplam_kayiplar"]) - 1
    print(
        f"{son_ep+1:<8} | "
        f"{egitim_raporu['toplam_kayiplar'][son_ep]:>10.4f} | "
        f"{egitim_raporu['kayiplar_sft'][son_ep]:>8.4f} | "
        f"{egitim_raporu['kayiplar_or'][son_ep]:>8.4f} | "
        f"{egitim_raporu['log_odds_oranlari'][son_ep]:>14.3f} | "
        f"%{egitim_raporu['dogruluklar'][son_ep]:>11.2f}"
    )
    print("-" * 95)

    kiyas_raporu = lab.mimari_kiyas_raporu()
    print("\n[-] PPO vs DPO vs ORPO MİMARİ KARŞILAŞTIRMASI:")
    print(f"  * PPO (RLHF) Boru Hattı : {kiyas_raporu['ppo_asamalar']} -> GPU'da {kiyas_raporu['ppo_gpu_model']} Model")
    print(f"  * DPO Boru Hattı        : {kiyas_raporu['dpo_asamalar']} -> GPU'da {kiyas_raporu['dpo_gpu_model']} Model")
    print(f"  * ORPO Boru Hattı       : {kiyas_raporu['orpo_asamalar']} -> GPU'da {kiyas_raporu['orpo_gpu_model']} Model!")
    print(f"  * VRAM Tasarrufu        : {kiyas_raporu['orpo_vram_tasarrufu_dpo_gore']} (PPO'ya göre {kiyas_raporu['orpo_vram_tasarrufu_ppo_gore']})")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli ORPO Hizalama Teşhis Panosu Çiziliyor...")
    gorsellestirici = ORPOGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "orpo_alignment_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        kiyas_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 112: Odds Ratio Preference Optimization (ORPO) Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
