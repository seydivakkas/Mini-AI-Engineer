"""
Day 242: Diffusion Policy (Robotik Manipülasyon ve Yörünge Üretimi) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from src.diffusion_policy_motoru import (
    DiffusionPolicyScheduler,
    DiffusionUNet1D,
    DiffusionPolicyController,
)
from src.diffusion_policy_profilleyici import DiffusionPolicyProfilleyici
from src.gorsellestirici import DiffusionPolicyGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 242 (FAZ 13): DIFFUSION POLICY — ROBOTİK MANİPÜLASYON VE YÖRÜNGE ÜRETİMİ İÇİN KOŞULLU DİFÜZYON")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: DDPM Zamanlayıcısı ve 1D U-Net Modelinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Diffusion Policy DDPM Zamanlayıcısı ve 1D U-Net Başlatılıyor...")
    scheduler = DiffusionPolicyScheduler(adim_sayisi=16)
    model = DiffusionUNet1D(eylem_boyutu=7, eylem_ufku=8, kosul_boyutu=64, gizli_boyut=128)
    toplam_parametre = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 1D Zamansal U-Net Modeli Hazır. Toplam Parametre: {toplam_parametre:,}")
    print(f"  ✓ Difüzyon Adım Sayısı K={scheduler.adim_sayisi}, Eylem Ufku Ta={model.eylem_ufku}")

    # -------------------------------------------------------------
    # ADIM 2: Çok Modlu Eylem Bloku Üretimi (Action Chunking)
    # -------------------------------------------------------------
    print("\n[2/4] Koşullandırılmış Difüzyon Eylem Bloku Üretiliyor...")
    controller = DiffusionPolicyController(model, scheduler)
    kosul_vektoru = torch.randn(1, 64)

    eylem_bloku = controller.eylem_bloku_uret(kosul_vektoru)
    print(f"  ✓ Üretilen Eylem Bloku Boyutu: {eylem_bloku.shape} (Ta=8 Adım, 7-DoF Eylem)")

    # -------------------------------------------------------------
    # ADIM 3: Kayan Ufuklu Kapalı Döngü İcra (Receding Horizon)
    # -------------------------------------------------------------
    print("\n[3/4] Kayan Ufuklu Kapalı Döngü İcra Ediliyor (Te=4 Adım)...")
    icra_edilen_adilar = controller.kayan_ufuk_icra_et(kosul_vektoru, icra_adimi=4)

    for i, adim in enumerate(icra_edilen_adilar, 1):
        print(f"  [İcra Adımı {i}] ΔHız: [Δx={adim[0]:.3f}, Δy={adim[1]:.3f}, Δz={adim[2]:.3f}] | Tutucu: {adim[6]:.2f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Diffusion Policy Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DiffusionPolicyProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "diffusion_policy_paneli.png")

    DiffusionPolicyGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Diffusion Policy Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 242 (FAZ 13): DIFFUSION POLICY ROBOTİK MANİPÜLASYON MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
