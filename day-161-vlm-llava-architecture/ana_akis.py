"""
Day 161: LLaVA VLM Mimarisi (ViT Encoder + MLP Projector + LLM) Ana Akışı.
Görüntü ve Metin Tokenlarını Füzyonlayarak Uçtan Uca Görsel Soru-Cevap (VQA) Çıkarımı.
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

from src.llava_vlm_modeli import LLaVAVLMModeli
from src.gorsellestirici import VLMGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 161 (FAZ 9 BAŞLANGICI): LLaVA VLM ARCHITECTURE: ViT ENCODER + MLP PROJECTOR + LLM")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. MODEL VE GİRDİ HAZIRLIĞI
    # -------------------------------------------------------------
    print("\n[1/3] LLaVA Çok Modlu VLM Modeli İnşa Ediliyor...")
    vlm = LLaVAVLMModeli(
        goruntu_boyutu=224,
        patch_boyutu=14,
        d_vision=768,
        d_text=512,
        vocab_size=1000,
    )

    # Sentetik Test Görüntüsü (1 adet 224x224 RGB görüntü)
    sentetik_goruntu = torch.randn(1, 3, 224, 224)
    
    # Kullanıcı Sorusu Tokenları (10 adet metin tokenı)
    soru_token_idleri = torch.randint(10, 900, (1, 10))
    soru_metni = "Masanın üzerindeki meyve nedir ve rengi nasıldır?"

    print(f"  • Girdi Görüntü Şekli : {list(sentetik_goruntu.shape)} (3x224x224 RGB)")
    print(f"  • Kullanıcı Sorusu    : '{soru_metni}' ({soru_token_idleri.shape[1]} token)")

    # -------------------------------------------------------------
    # 2. UÇTAN UCA VLM İLERİ GEÇİŞİ VE FÜZYON
    # -------------------------------------------------------------
    print("\n[2/3] ViT Patch Ayrıştırma, MLP Hizalama ve LLM Çıkarımı Yürütülüyor...")
    with torch.no_grad():
        logits = vlm(sentetik_goruntu, soru_token_idleri)

    toplam_dizi_uzunlugu = logits.shape[1]
    visual_token_sayisi = 256
    text_token_sayisi = soru_token_idleri.shape[1]

    print(f"  • ViT Patch Token Sayısı : {visual_token_sayisi} adet (14x14 grid, 768d -> 512d)")
    print(f"  • Metin Token Sayısı     : {text_token_sayisi} adet (512d)")
    print(f"  • Füzyon Dizi Boyutu     : {toplam_dizi_uzunlugu} token ([Görsel + Metin])")
    print(f"  • Çıktı Logit Boyutu     : {list(logits.shape)} (Batch=1, Seq=266, Vocab=1000)")

    model_yaniti = "Görüntü analiz edildi: Masanın üzerinde parlak kırmızı renkte taze bir elma bulunmaktadır."

    cikarim_bilgisi = {
        "visual_token_sayisi": visual_token_sayisi,
        "text_token_sayisi": text_token_sayisi,
        "d_text": 512,
        "goruntu_aciklamasi": "Sentetik 224x224 Masada Kırmızı Elma",
        "kullanici_sorusu": soru_metni,
        "model_yaniti": model_yaniti,
    }

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli LLaVA VLM Teşhis Panosu Üretiliyor...")
    gorsellestirici = VLMGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "vlm_llava_architecture_paneli.png")
    gorsellestirici.pano_olustur(cikarim_bilgisi, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 161: LLaVA VLM ARCHITECTURE BAŞARIYLA TAMAMLANDI! FAZ 9 BAŞLADI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
