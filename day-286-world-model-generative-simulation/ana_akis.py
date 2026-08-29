"""
Day 286 (FAZ 15): Dünya Modelleri ve Üretken Simülasyon (DreamerV3) Ana Akış Betiği.
RSSM Durum Uzayı, Gizil Hayal Gücü (Latent Imagination) ve 100x Örnek Verimliliği.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.world_model_motoru import RSSMCell, WorldModelEngine
from src.world_model_profilleyici import WorldModelProfilleyici
from src.gorsellestirici import WorldModelGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 286 (FAZ 15): DÜNYA MODELLERİ VE ÜRETKEN SİMÜLASYON — DREAMERV3 & RSSM")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: RSSM Çekirdeğinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] RSSM (Recurrent State-Space Model) Çekirdeği Başlatılıyor...")
    rssm = RSSMCell(action_dim=2, deter_dim=64, stoch_dim=16)
    print(f"  • Deterministik Durum Boyutu (h_t)   : 64 Birim (GRU Cell)")
    print(f"  • Stokastik Gizil Durum Boyutu (z_t) : 16 Boyut (Gaussian Dağılım)")
    print(f"  • Birleşik Durum Temsili             : [h_t, z_t] -> 80 Boyut")

    # -------------------------------------------------------------
    # ADIM 2: Gizil Hayal Gücü Simülasyonu (Horizon H=15)
    # -------------------------------------------------------------
    print("\n[2/4] Gizil Hayal Gücü Simülasyonu Çalıştırılıyor (Latent Imagination Horizon H=15)...")
    initial_h = torch.zeros(1, 64)
    initial_z = torch.randn(1, 16)

    imag_res = WorldModelEngine.simulate_latent_imagination(
        rssm=rssm,
        initial_h=initial_h,
        initial_z=initial_z,
        horizon=15,
    )
    print(f"  • Simüle Edilen Gelecek Adımı (H)   : {imag_res['horizon']} Adım İleri")
    print(f"  • Gerçek Çevre Etkileşimi            : 0 Adım (Tamamen İç Hayal Gücünde)")
    print(f"  • Hayal Edilen Kümülatif Ödül        : {imag_res['total_imagined_reward']:.2f}")

    # -------------------------------------------------------------
    # ADIM 3: Model-Free RL vs DreamerV3 Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Model-Free RL vs DreamerV3 Dünya Modeli Kıyaslama Raporu...")
    profil = WorldModelProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Model-Free RL (PPO)                : 1,000,000 Gerçek Adım | Nihai Skor: {kars['nihai_epizodik_odul']['Model_Free_PPO']:.1f}")
    print(f"  • Model-Based (MBPO)                 : 250,000 Gerçek Adım   | Nihai Skor: {kars['nihai_epizodik_odul']['Model_Based_MBPO']:.1f}")
    print(f"  • Dünya Modeli (DreamerV3)           : 10,000 Gerçek Adım    | Nihai Skor: {kars['nihai_epizodik_odul']['DreamerV3_WorldModel']:.1f}")
    print(f"  • Örnek Verimlilik Kazancı           : {profil['ornek_verimlilik_kazanci']:.0f}x Daha Az Gerçek Çevre Etkileşimi")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Dünya Modelleri (DreamerV3) Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "world_model_dreamerv3_paneli.png")

    WorldModelGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Dünya Modelleri Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 286 (FAZ 15): DÜNYA MODELLERİ VE ÜRETKEN SİMÜLASYON MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
