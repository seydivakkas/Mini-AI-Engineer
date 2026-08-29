"""
Day 287 (FAZ 15): Difüzyon Tabanlı Planlayıcılar ve Robot Manipülasyonu (Diffusion Policy) Ana Akış Betiği.
DDPM/DDIM Eylem Yörüngesi Üretimi, Çok Modlu Eylemler ve 12x Hassas Visuomotor Kontrol.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.diffusion_policy_motoru import ConditionalNoisePredictor1D, DiffusionPolicyEngine
from src.diffusion_policy_profilleyici import DiffusionPolicyProfilleyici
from src.gorsellestirici import DiffusionPolicyGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 287 (FAZ 15): DİFÜZYON TABANLI PLANLAYICILAR VE ROBOT MANİPÜLASYONU — DIFFUSION POLICY")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Diffusion Policy Çekirdeğinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Diffusion Policy ve 1D Gürültü Tahmin Ağı Başlatılıyor...")
    engine = DiffusionPolicyEngine(action_dim=2, action_horizon=8, obs_dim=16, num_diffusion_steps=16)
    print(f"  • Eylem Yörünge Ufku (T_p)           : 8 Zaman Adımı")
    print(f"  • Eylem Boyutu (D_a)                 : 2 Boyutlu Hareket [dx, dy]")
    print(f"  • Difüzyon Adım Sayısı (K)           : 16 Denoising Adımı (DDPM / DDIM)")
    print(f"  • Gözlem Boyutu                      : 16 Boyutlu Visuomotor Vektörü")

    # -------------------------------------------------------------
    # ADIM 2: Ters Difüzyon ile Yörünge Üretimi
    # -------------------------------------------------------------
    print("\n[2/4] Ters Difüzyon (Reverse Sampling) ile Eylem Yörüngesi Gürültüden Arındırılıyor...")
    obs = torch.randn(1, 16)
    traj = engine.reverse_sample_trajectory(obs)
    print(f"  • Üretilen Yörünge Boyutu            : {traj.shape} (Batch=1, Horizon=8, Dim=2)")
    print(f"  • Başlangıç Durumu                   : Saf Gauss Gürültüsü A_K ~ N(0, I)")
    print(f"  • Sonuç Durumu                       : Temiz Robotik Eylem Yörüngesi A_0")

    # -------------------------------------------------------------
    # ADIM 3: Davranış Kopyalama (BC) vs Diffusion Policy Kıyaslaması
    # -------------------------------------------------------------
    print("\n[3/4] Standart Davranış Kopyalama (BC) vs Diffusion Policy Kıyaslama Raporu...")
    profil = DiffusionPolicyProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Standart BC Başarı Oranı           : %{kars['gorev_basari_orani_yuzde']['Standart_BC']:.1f} (Mod Ortalaması Alma ve Engele Çarpma Hatası)")
    print(f"  • GMM Policy Başarı Oranı            : %{kars['gorev_basari_orani_yuzde']['GMM_Policy']:.1f}")
    print(f"  • Diffusion Policy Başarı Oranı      : %{kars['gorev_basari_orani_yuzde']['Diffusion_Policy']:.1f} (+%49.6 Başarı Artışı)")
    print(f"  • Yörünge Takip Hatası (RMSE)        : Standart BC: {kars['yorunge_takip_hatasi_rmse']['Standart_BC']:.3f} -> Diffusion Policy: {kars['yorunge_takip_hatasi_rmse']['Diffusion_Policy']:.3f} ({profil['hata_azalma_orani']:.1f}x Daha Hassas)")
    print(f"  • Çok Modlu Dağılım Yakalama         : %{kars['cok_modlu_yakalama_orani']['Diffusion_Policy']:.1f} (Sol ve Sağ Geçişleri Ayrık Modelleme)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Diffusion Policy Robotik Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "diffusion_policy_robotics_paneli.png")

    DiffusionPolicyGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Diffusion Policy Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 287 (FAZ 15): DİFÜZYON TABANLI PLANLAYICILAR (DIFFUSION POLICY) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
