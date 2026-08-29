"""
Day 114: Group Relative Policy Optimization (GRPO - DeepSeek-R1) ile Akıl Yürütme Ana Akışı.
Critic'siz grup göreli avantaj normalizasyonu (Z-Score), kural tabanlı doğruluk/düşünme ödülleri ve 6 panelli teşhis panosu.
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

from src.grpo_laboratuvari import GRPOLaboratuvari
from src.gorsellestirici import GRPOGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 114: Group Relative Policy Optimization (GRPO - DeepSeek-R1) Reasoning Alignment")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Model ve Laboratuvar Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] GRPO Politika ve Referans Modelleri Başlatılıyor...")
    lab = GRPOLaboratuvari(
        vocab_size=1000,
        dim=256,
        num_heads=4,
        num_layers=4,
        cihaz=cihaz,
    )

    model_p = sum(p.numel() for p in lab.model.parameters())
    print(f"  * 1. Politika Modeli (pi_theta) : {model_p:,} Parametre (Eğitilebilir)")
    print(f"  * 2. Referans Modeli (pi_ref)   : {model_p:,} Parametre (Dondurulmuş - D_KL için)")
    print("  * 3. CRITIC (Value Network V)   : GEREK YOK (0 Parametre - %65 VRAM Tasarrufu!)")
    print("  * 4. Grup Örnekleme Boyutu (G)  : G = 8 Paralel Çıktı / Prompt")

    # -------------------------------------------------------------
    # ADIM 2: GRPO Akıl Yürütme Eğitimi (15 Epok, G=8)
    # -------------------------------------------------------------
    print("\n[2/3] GRPO Akıl Yürütme Eğitimi Koşturuluyor (15 Epok, G=8 Grup Rollout)...")
    egitim_raporu = lab.grpo_egit(
        prompt_sayisi=35,
        group_size=8,
        epok_sayisi=15,
        lr=5e-4,
    )

    print("\n--- GRPO AKIL YÜRÜTME & GRUP AVANTAJ GELİŞİMİ ---")
    print(f"{'EPOK':<8} | {'TOPLAM LOSS':<12} | {'POLİTİKA KAYBI':<15} | {'KL KAYBI':<10} | {'ORTALAMA ÖDÜL':<15} | {'ÖDÜL STD (σ)':<14} | {'KIRPILMA (%)':<12}")
    print("-" * 95)
    for ep in range(0, len(egitim_raporu["toplam_kayiplar"]), 3):
        print(
            f"{ep+1:<8} | "
            f"{egitim_raporu['toplam_kayiplar'][ep]:>10.4f} | "
            f"{egitim_raporu['politika_kayiplari'][ep]:>13.4f} | "
            f"{egitim_raporu['kl_kayiplari'][ep]:>8.4f} | "
            f"{egitim_raporu['ortalama_oduller'][ep]:>13.3f} | "
            f"{egitim_raporu['std_oduller'][ep]:>12.3f} | "
            f"%{egitim_raporu['kirpilma_oranlari'][ep]:>10.2f}"
        )
    # Son epok
    son_ep = len(egitim_raporu["toplam_kayiplar"]) - 1
    print(
        f"{son_ep+1:<8} | "
        f"{egitim_raporu['toplam_kayiplar'][son_ep]:>10.4f} | "
        f"{egitim_raporu['politika_kayiplari'][son_ep]:>13.4f} | "
        f"{egitim_raporu['kl_kayiplari'][son_ep]:>8.4f} | "
        f"{egitim_raporu['ortalama_oduller'][son_ep]:>13.3f} | "
        f"{egitim_raporu['std_oduller'][son_ep]:>12.3f} | "
        f"%{egitim_raporu['kirpilma_oranlari'][son_ep]:>10.2f}"
    )
    print("-" * 95)

    kiyas_raporu = lab.ppo_vs_grpo_kiyas_raporu()
    print("\n[-] PPO vs GRPO (DeepSeek-R1) MİMARİ KARŞILAŞTIRMASI:")
    for i, krt in enumerate(kiyas_raporu["kriterler"]):
        print(f"  * {krt:<25}: PPO -> {kiyas_raporu['ppo'][i]:<25} | GRPO -> {kiyas_raporu['grpo'][i]}")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli GRPO Akıl Yürütme Teşhis Panosu Çiziliyor...")
    gorsellestirici = GRPOGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "grpo_reasoning_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        kiyas_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 114: Group Relative Policy Optimization (GRPO) Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
