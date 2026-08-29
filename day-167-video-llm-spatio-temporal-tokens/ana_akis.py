"""
Day 167: Video LLM Spatio-Temporal Token Modelleme Ana Akışı (FAZ 9).
Zamansal Kare Örnekleme, 3D Space-Time Dikkat Çıkarımı ve Video-QA Anlatımı.
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

from src.zamansal_ornekleyici import ZamansalKareOrnekleyici
from src.video_llava_modeli import VideoLLaVAModeli
from src.gorsellestirici import VideoLLMGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 167 (FAZ 9): VIDEO LLM: SPATIO-TEMPORAL TOKENS & 3D SPACE-TIME ATTENTION")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. ZAMANSAL KARE ÖRNEKLEME SİMÜLASYONU
    # -------------------------------------------------------------
    print("\n[1/3] 60 Karelik Ham Videodan 8 Kare Zamansal Örnekleme Yapılıyor...")
    uniform_kareler = ZamansalKareOrnekleyici.duzenli_ornekle(toplam_kare=60, ornek_sayisi=8)
    print(f"  • Uniform (Eşit Aralıklı) Kare İndeksleri: {uniform_kareler}")

    hareket_skorlari = [0.1] * 60
    hareket_skorlari[14] = 0.95  # Kedi koşmaya başladı
    hareket_skorlari[28] = 0.88  # Kedi zıpladı
    hareket_skorlari[42] = 0.92  # Koltuğa kondu
    adaptive_kareler = ZamansalKareOrnekleyici.uyarlamali_ornekle(hareket_skorlari, ornek_sayisi=8)
    print(f"  • Adaptive (Harekete Duyarlı) Kareler : {adaptive_kareler}")

    # -------------------------------------------------------------
    # 2. VIDEO-LLaVA MODEL ÇIKARIMI
    # -------------------------------------------------------------
    print("\n[2/3] Video-LLaVA Spatio-Temporal Projektör ve LLM Girdisi Üretiliyor...")
    model = VideoLLaVAModeli(kare_sayisi=8, kare_basina_token=16, viz_dim=256, llm_dim=512)
    ornek_video = torch.randn(1, 8, 16, 256)
    llm_video_tokens = model(ornek_video)
    print(f"  • Ham Video Tensor Boyutu       : {list(ornek_video.shape)} [B, T, N, D]")
    print(f"  • LLM'e Giren Video Token Sayısı: {list(llm_video_tokens.shape)} [B, Total_Tokens, LLM_Dim]")

    rapor = VideoLLaVAModeli.ornek_video_qa_senaryolarini_degerlendir()
    print("\n" + "-" * 90)
    print(f"{'Video Dosyası':<25} | {'Soru':<35} | {'Doğruluk'}")
    print("-" * 90)
    for s in rapor["senaryolar"]:
        print(f"{s['video_adi']:<25} | {s['soru'][:33]+'..':<35} | %{s['dogruluk_skoru']*100:.1f}")
    print("-" * 90)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Video LLM Teşhis Panosu Üretiliyor...")
    gorsellestirici = VideoLLMGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "video_llm_spatio_temporal_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 167: VIDEO LLM SPATIO-TEMPORAL TOKENS BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
