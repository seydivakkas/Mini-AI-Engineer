"""
Day 255: Teleoperasyon ve Taklit Öğrenmesi (ACT & Behavior Cloning) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from src.act_imitation_motoru import (
    TeleoperationDataBuffer,
    ACTCVAEModel,
    TemporalEnsembler,
)
from src.act_imitation_profilleyici import ACTImitationProfilleyici
from src.gorsellestirici import ACTImitationGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 255 (FAZ 13): TELEOPERASYON VE TAKLİT ÖĞRENMESİ (BEHAVIOR CLONING & ACT)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Teleoperasyon Veri Havuzunun Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] İnsan Teleoperasyon Veri Havuzu (Data Buffer) Yükleniyor...")
    buffer = TeleoperationDataBuffer(chunk_size=10)

    # 3 Adet Örnek İnsan Teleoperasyon Demosu Ekleyelim
    for ep in range(1, 4):
        demo_states = np.random.randn(40, 14)
        demo_actions = np.random.randn(40, 7)
        buffer.add_demonstration(demo_states, demo_actions)

    print(f"  • Yüklenen Örnek Demo Sayısı: 3 Adet (Toplam {len(buffer.samples)} Dilimlenmiş Chunk)")
    print(f"  • Eylem Yığını Boyutu (K)   : {buffer.chunk_size} Adım")

    # -------------------------------------------------------------
    # ADIM 2: ACT CVAE ve Transformer Modelinin Kurulması
    # -------------------------------------------------------------
    print("\n[2/4] Action Chunking with Transformers (ACT) CVAE Modeli Başlatılıyor...")
    model = ACTCVAEModel(state_dim=14, action_dim=7, chunk_size=10, latent_dim=16)
    ensembler = TemporalEnsembler(chunk_size=10, m_decay=0.05)

    print(f"  • Giriş Durum Boyutu        : 14D (Çoklu Kamera + Eklem Açıları)")
    print(f"  • Latent Niyet Boyutu (z)   : {model.latent_dim}D (CVAE Stili)")
    print(f"  • Çıktı Eylem Yığını (A_t)  : [{model.chunk_size}, {model.action_dim}] (7-DoF Kol)")

    # -------------------------------------------------------------
    # ADIM 3: Zamansal Topluluk (Temporal Ensembling) ile Çıkarım
    # -------------------------------------------------------------
    print("\n[3/4] Zamansal Topluluk ile Pürüzsüz Eylem Çıkarımı İcra Ediliyor...")
    current_state = np.random.randn(14).astype(np.float32)

    with torch.no_grad():
        s_tensor = torch.tensor(current_state).unsqueeze(0)
        pred_chunk, _, _ = model(s_tensor)
        pred_chunk_np = pred_chunk.squeeze(0).numpy()

    # 4 Zaman Adımı Boyunca Çakışan Tahminleri Birleştir
    for t_step in range(1, 5):
        # Gerçek zamanlı küçük durum gürültüsü ile yeni chunk tahmini
        sim_pred = pred_chunk_np + np.random.randn(*pred_chunk_np.shape) * 0.02
        ensembler.add_prediction(sim_pred)
        ensembled_act = ensembler.get_ensembled_action()
        print(f"  [Adım {t_step}] Ensembled Eylem (7-DoF): {ensembled_act.tolist()}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli ACT Teleoperasyon Teşhis Panosu Oluşturuluyor...")
    profil_raporu = ACTImitationProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "act_teleoperation_paneli.png")

    ACTImitationGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ ACT Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 255 (FAZ 13): TELEOPERASYON VE TAKLİT ÖĞRENMESİ (ACT) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
