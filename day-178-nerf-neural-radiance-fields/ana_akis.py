"""
Day 178: NeRF (Neural Radiance Fields) 3D Sahne Hacimsel Sentezi Ana Akışı (FAZ 9).
Işın Takibi (Ray Marching), Hacimsel Render ve 6 Panelli Teşhis Panosu.
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

from src.nerf_mlp import NeRFModeli
from src.hacimsel_isin_izleyici import HacimselIsinIzleyici
from src.gorsellestirici import NeRFGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 178 (FAZ 9): NEURAL RADIANCE FIELDS (NeRF): NOVEL VIEW SYNTHESIS & VOLUMETRIC RENDERING")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. NERF MODELİ VE IŞIN İZLEME BAŞLATILIYOR
    # -------------------------------------------------------------
    print("\n[1/3] NeRF Hacimsel MLP Modeli ve Işın İzleyici Başlatılıyor...")
    model = NeRFModeli(pos_frequencies=10, dir_frequencies=4, hidden_dim=128)
    renderer = HacimselIsinIzleyici(model, near=2.0, far=6.0, num_samples=64)

    # 4 Adet Kamera Işını Simülasyonu
    rays_o = torch.zeros(4, 3)  # Kamera merkezi [0, 0, 0]
    rays_d = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.99],
        [-0.1, 0.0, 0.99],
        [0.0, 0.1, 0.99],
    ])
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    render_sonucu = renderer.render_isin(rays_o, rays_d, perturb=False)

    print(f"  • Simüle Edilen Kamera Işını Sayısı : {rays_o.shape[0]}")
    print(f"  • Işın Başına Örnekleme Noktası    : {renderer.num_samples} (Tabakalı / Stratified)")
    print(f"  • Üretilen Piksel RGB Renkleri     :\n{render_sonucu['rgb']}")
    print(f"  • Hesaplanan Hacimsel Derinlik     : {render_sonucu['depth'].tolist()} metre")

    # -------------------------------------------------------------
    # 2. 3D SAHNE TEMSİLİ VE PSNR ANALİZİ
    # -------------------------------------------------------------
    print("\n[2/3] 3D Sahne Temsili ve Kalite Analizi Yapılıyor...")
    rapor = HacimselIsinIzleyici.ornek_nerf_sahne_raporu()

    print(f"\n>> SAHNE: {rapor['sahne_adi']}")
    print("-" * 88)
    print(f"{'3D Temsil Yöntemi':<35} | {'PSNR (Kalite)':<15} | {'Bellek':<12} | {'Açıklama'}")
    print("-" * 88)
    for k in rapor["karsilastirma"]:
        print(f"{k['yontem']:<35} | {k['psnr']} dB{'':<7} | {k['bellek_mb']} MB{'':<5} | {k['durum']}")
    print("-" * 88)
    print(f"Fourier Frekans Seviyeleri: {rapor['fourier_l_seviyesi']}")
    print(f"SSIM Benzerlik Skoru     : %{rapor['metrikler']['ssim']*100:.1f}")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli NeRF Teşhis Panosu Üretiliyor...")
    gorsellestirici = NeRFGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "nerf_neural_radiance_fields_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 178: NEURAL RADIANCE FIELDS (NeRF) BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
