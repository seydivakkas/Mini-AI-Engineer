"""
Day 110: Direct Preference Optimization (DPO) ile LLM Tercih Hizalaması Ana Akışı.
Reward modelsiz doğrudan log-oranı optimizasyonu, örtük ödül ayrışması ve 6 panelli teşhis panosu.
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

from src.dpo_laboratuvari import DPOLaboratuvari
from src.gorsellestirici import DPOGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 110: Direct Preference Optimization (DPO) & Implicit Reward Alignment")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Model ve Sentetik Tercih Veri Seti Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] DPO Tercih Veri Seti ve Dil Modelleri Başlatılıyor...")
    lab = DPOLaboratuvari(
        vocab_size=1000,
        dim=256,
        num_heads=4,
        num_layers=4,
        cihaz=cihaz,
    )
    c_ids, r_ids, c_mask, r_mask = lab.sentetik_tercih_verisi_uret(
        cift_sayisi=350, prompt_len=10, resp_len=14
    )

    policy_p = sum(p.numel() for p in lab.policy_model.parameters())
    print(f"  * Üretilen Tercih Çifti Sayısı  : {c_ids.shape[0]}")
    print(f"  * Dizi Uzunluğu (Prompt+Yanıt)  : {c_ids.shape[1]} Token")
    print(f"  * 1. Politika Modeli (pi_theta) : {policy_p:,} Parametre (Eğitilebilir)")
    print(f"  * 2. Referans Modeli (pi_ref)   : {policy_p:,} Parametre (Dondurulmuş)")
    print("  * 3. Critic & Reward Modeli     : GEREK YOK (0 Parametre - %50 VRAM Tasarrufu!)")

    # -------------------------------------------------------------
    # ADIM 2: DPO Tercih Eğitimi (20 Epok, Beta=0.1)
    # -------------------------------------------------------------
    print("\n[2/3] DPO Doğrudan Tercih Optimizasyonu Koşturuluyor (20 Epok, Beta=0.1)...")
    egitim_raporu = lab.dpo_egit(
        chosen_ids=c_ids,
        rejected_ids=r_ids,
        chosen_mask=c_mask,
        rejected_mask=r_mask,
        epok_sayisi=20,
        batch_size=32,
        lr=1e-3,
        beta=0.1,
    )

    print("\n--- DPO TERCİH HİZALAMA EĞİTİM GELİŞİMİ ---")
    print(f"{'EPOK':<8} | {'DPO KAYBI':<12} | {'DOĞRULUK (%)':<16} | {'ÖRTÜK r_w':<12} | {'ÖRTÜK r_l':<12} | {'MARJİN (Δr)':<14}")
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

    kiyas_raporu = lab.dpo_vs_ppo_kiyasi()
    print("\n[-] DPO vs PPO MİMARİ VE VERİMLİLİK ÖZETİ:")
    print(f"  * PPO Eşzamanlı Model Sayısı   : {kiyas_raporu['ppo_model_sayisi']} (Actor, Critic, Ref, RM)")
    print(f"  * DPO Eşzamanlı Model Sayısı   : {kiyas_raporu['dpo_model_sayisi']} (Policy, Ref)")
    print(f"  * VRAM ve Bellek Tasarrufu     : %{kiyas_raporu['vram_tasarrufu_yuzde']:.1f} Kazanç")
    print(f"  * Örnekleme / Rollout İhtiyacı : {kiyas_raporu['ornekleme_gereksinimi']}")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli DPO Hizalama Teşhis Panosu Çiziliyor...")
    gorsellestirici = DPOGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "dpo_alignment_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        kiyas_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 110: Direct Preference Optimization (DPO) Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
