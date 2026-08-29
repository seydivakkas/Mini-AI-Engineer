"""
Day 179: 3D Gaussian Splatting (3DGS) Ana Akış ve Teşhis Panosu İcra Motoru.
Kerbl et al. (2023) Gerçek Zamanlı (100+ FPS) 3D Radyan ve Nokta Kümesi Renderı.
"""

import sys
import os
import torch
import numpy as np

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Modül yolunu ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gaussian_temsili import Gaussian3D
from src.diferansiyellenebilir_rasterizer import GaussianRasterizer
from src.gorsellestirici import GaussianGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 179 (FAZ 9): 3D GAUSSIAN SPLATTING (3DGS): REAL-TIME 100+ FPS RADIANCE FIELD RENDERING")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # 1. 3D Gauss Nokta Kümesini Başlat
    num_gaussians = 150
    torch.manual_seed(42)
    gaussians = Gaussian3D(num_gaussians=num_gaussians)

    print(f"\n[1/4] 3D Gauss Nokta Kümesi Oluşturuldu:")
    print(f"  • Toplam Gauss Elipsoidi : {num_gaussians} adet")
    print(f"  • Konum (mu) Şekli       : {gaussians.mu.shape} (R^3 uzayında)")
    print(f"  • Ölçek (scale) Şekli    : {gaussians.get_scaling().shape} (Pozitif exp)")
    print(f"  • Dönme Kuaterniyonu (q) : {gaussians.rotation.shape} (Birim kuaterniyon)")
    print(f"  • Opaklık (alpha)        : {gaussians.get_opacity().shape} (Sigmoid [0, 1])")

    # 2. Kamera ve Bakış Açısı Matrisleri (View Matrix)
    # Kamera [0, 0, 2.5] konumunda sahneye bakıyor
    R_cam = torch.eye(3, dtype=torch.float32)
    T_cam = torch.tensor([0.0, 0.0, 2.5], dtype=torch.float32)

    # 3. Diferansiyellenebilir Rasterizer ile Render Al
    rasterizer = GaussianRasterizer(width=64, height=64)
    print(f"\n[2/4] Diferansiyellenebilir Tile-Tabanlı Alfa Rasterizasyonu Başlatılıyor...")
    render_out = rasterizer.render(
        gaussians=gaussians,
        view_matrix_R=R_cam,
        view_matrix_T=T_cam,
        fx=70.0,
        fy=70.0,
        bg_color=(0.05, 0.05, 0.08),
    )

    rendered_img = render_out["image"]
    print(f"  ✓ Render Başarılı!")
    print(f"  • Çıkış Görüntüsü Boyutu : {rendered_img.shape} [Yükseklik x Genişlik x RGB]")
    print(f"  • Render Edilen Gausslar : {render_out['num_rendered']} adet (Pozitif Derinlikte)")
    print(f"  • Ortalama Piksel Değeri : {rendered_img.mean().item():.4f}")

    # 4. Kıyaslama Raporu ve 6 Panelli Görselleştirme
    print(f"\n[3/4] NeRF vs 3DGS Performans ve Hız Kıyaslama Raporu Derleniyor...")
    rapor = rasterizer.ornek_3dgs_kiyaslama_raporu()
    for item in rapor["karsilastirma"]:
        print(f"  • {item['yontem']:<35} : {item['fps']:>6.1f} FPS | {item['psnr']:>4.1f} dB PSNR | Tip: {item['tip']}")
    print(f"  >> HIZ ARTIŞI: {rapor['fps_artis_kati']}")

    # 5. Teşhis Panosu Üretimi
    print(f"\n[4/4] 6 Panelli 3DGS Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = GaussianGorsellestirici(dpi=300)
    mu_3d_np = gaussians.mu.detach().cpu().numpy()
    cikti_resmi = os.path.join(cikis_dizini, "gaussian_splatting_3dgs_paneli.png")
    gorsellestirici.pano_olustur(
        rapor=rapor,
        rendered_data=render_out,
        mu_3d=mu_3d_np,
        kayit_yolu=cikti_resmi,
    )

    print("\n" + "=" * 110)
    print("✓ Day 179: 3D GAUSSIAN SPLATTING (3DGS) BAŞARIYLA TAMAMLANDI! (100+ FPS RENDER)")
    print("=" * 110)


if __name__ == "__main__":
    main()
