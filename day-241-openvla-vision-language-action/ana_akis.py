"""
Day 241: OpenVLA (Vision-Language-Action) Robotik Mimari Ana Akışı (FAZ 13 BAŞLANGICI).
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from src.openvla_motoru import (
    OpenVLAActionTokenizer,
    OpenVLAModel,
    OpenVLAController,
)
from src.openvla_profilleyici import OpenVLAProfilleyici
from src.gorsellestirici import OpenVLAGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 241 (FAZ 13 BAŞLANGICI): OPENVLA — VISION-LANGUAGE-ACTION (VLA) ROBOTİK MANİPÜLASYON MİMARİSİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 7-DoF Eylem Belirteçleyicisinin Test Edilmesi
    # -------------------------------------------------------------
    print("\n[1/4] 7-DoF Eylem Belirteçleyicisi Başlatılıyor (256 Ayrık Kova)...")
    tokenizer = OpenVLAActionTokenizer(kova_sayisi=256, eylem_boyutu=7)
    ornek_eylem = np.array([0.25, -0.50, 0.75, 0.00, 0.10, -0.20, 1.00], dtype=np.float32)
    belirtecler = tokenizer.tokenize_action(ornek_eylem)
    cozulmus_eylem = tokenizer.detokenize_action(belirtecler)

    print(f"  • Orijinal Sürekli Eylem [-1.0, 1.0] : {ornek_eylem.tolist()}")
    print(f"  • Ayrıklaştırılmış Belirteçler [0, 255]: {belirtecler.tolist()}")
    print(f"  • Geri Çözülmüş Sürekli Eylem         : {cozulmus_eylem.tolist()}")

    # -------------------------------------------------------------
    # ADIM 2: OpenVLA Modelinin Yüklenmesi
    # -------------------------------------------------------------
    print("\n[2/4] OpenVLA Görsel-Dil-Eylem Modeli Yükleniyor...")
    torch.manual_seed(42)
    model = OpenVLAModel(viz_dim=128, text_dim=128, gizli_boyut=256, eylem_sayisi=7, kova_sayisi=256)
    toplam_parametre = sum(p.numel() for p in model.parameters())
    print(f"  ✓ OpenVLA Modeli Hazır. Toplam Parametre: {toplam_parametre:,}")

    # -------------------------------------------------------------
    # ADIM 3: Robotik Yörünge İcrası ve Eylem Tahmini
    # -------------------------------------------------------------
    komut_metni = "Sarı fincanı al ve yeşil tabağın üzerine yerleştir"
    print(f"\n[3/4] Robotik Yörünge İcrası Başlatılıyor...")
    print(f"  • Doğal Dil Komutu : '{komut_metni}'")
    print(f"  • Görsel Gözlem    : [224x224 RGB Kamera Akışı]")

    controller = OpenVLAController(model)
    img_tensor = torch.randn(1, 128)
    text_tensor = torch.randn(1, 128)

    print("\n--- [Robotik Kontrol Döngüsü: 5 Adımlı Yörünge] ---")
    for adim in range(1, 6):
        delta, yeni_pos = controller.adim_yurut(img_tensor, text_tensor)
        tutucu_durumu = "KAPALI" if yeni_pos[6] > 0.5 else "AÇIK"
        print(f"  [Adım {adim}] ΔEylem: [Δx={delta[0]:.2f}, Δy={delta[1]:.2f}, Δz={delta[2]:.2f}] -> Robot Konum: [x={yeni_pos[0]:.3f}, y={yeni_pos[1]:.3f}, z={yeni_pos[2]:.3f}] | Tutucu: {tutucu_durumu}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli OpenVLA Teşhis Panosu Oluşturuluyor...")
    profil_raporu = OpenVLAProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "openvla_paneli.png")

    OpenVLAGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ OpenVLA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 241 (FAZ 13 BAŞLANGICI): OPENVLA VLA ROBOTİK MİMARİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
