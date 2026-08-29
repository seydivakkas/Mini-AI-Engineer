"""
Day 172: Latent Diffusion Modelleri (LDM / Stable Diffusion) Ana Akışı (FAZ 9).
VAE Gizli Uzayında İleri/Geri Difüzyon, Gürültü Zaman Çizelgesi ve Denoising UNet.
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

from src.gurultu_zaman_cizelgesi import GurultuZamanCizelgesi
from src.denoising_unet import DenoisingUNet
from src.latent_diffusion_motoru import LatentDiffusionMotoru
from src.gorsellestirici import LDMGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 172 (FAZ 9): LATENT DIFFUSION MODELS (LDM): VAE FORWARD/REVERSE DIFFUSION & DENOISING UNET")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. GÜRÜLTÜ ZAMAN ÇİZELGESİ VE İLERİ DİFÜZYON
    # -------------------------------------------------------------
    print("\n[1/3] Difüzyon Zaman Çizelgesi (Cosine Schedule) Başlatılıyor...")
    schedule = GurultuZamanCizelgesi(num_timesteps=1000, schedule_type="cosine")
    z_0 = torch.randn(1, 4, 16, 16)  # VAE gizli temsili
    t_ornek = torch.tensor([500])

    z_t, gurultu = schedule.ileri_difuzyon(z_0, t_ornek)
    print(f"  • Temiz Gizli Vektör (z_0)     : {list(z_0.shape)} [Batch=1, Channels=4, 16x16]")
    print(f"  • t={t_ornek.item()} Anındaki Gürültülü z_t: {list(z_t.shape)}")
    print(f"  • Kalan Sinyal Gücü (alpha_bar): {schedule.alphas_cumprod[t_ornek.item()].item():.4f}")

    # -------------------------------------------------------------
    # 2. DENOISING UNET VE GÜRÜLTÜ KESTİRİM KAYBI
    # -------------------------------------------------------------
    print("\n[2/3] Denoising UNet ile VAE Gizli Uzayında Gürültü Kestirimi Yapılıyor...")
    unet = DenoisingUNet(in_channels=4, out_channels=4, base_channels=32)
    motor = LatentDiffusionMotoru(unet=unet, schedule=schedule)

    kayip, _, eps_pred = motor.kayip_hesapla(z_0)
    print(f"  • Kestirilen Gürültü (eps_pred): {list(eps_pred.shape)}")
    print(f"  • Difüzyon Kestirim Kaybı (MSE): {kayip.item():.6f}")

    rapor = LatentDiffusionMotoru.ornek_difuzyon_senaryolarini_getir()
    print("\n" + "-" * 80)
    print(f"{'Metrik':<40} | {'Değer'}")
    print("-" * 80)
    print(f"{'Toplam Difüzyon Adım Sayısı (T)':<40} | {rapor['num_timesteps']}")
    print(f"{'VAE Çözünürlük Sıkıştırma Oranı':<40} | {rapor['vae_sikistirma_orani']}")
    print(f"{'Hesaplama ve Bellek Tasarrufu':<40} | {rapor['hesaplama_tasarrufu']}")
    print(f"{'Ortalama Gürültü Kestirim Hatası':<40} | MSE = {rapor['ortalama_gurultu_kestirim_mse']:.4f}")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Latent Diffusion Teşhis Panosu Üretiliyor...")
    gorsellestirici = LDMGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "latent_diffusion_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 172: LATENT DIFFUSION MODELS (LDM) BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
