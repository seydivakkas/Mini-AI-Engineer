"""
Day 171: Uçtan Uca Speech-to-Speech LLM (DuaLLM / Moshi) Ana Akışı (FAZ 9).
Doğrudan Ses Tokenı Alıp Ses Tokenı Üretme, Çift Başlıklı Mimari ve Gecikme Analizi.
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

from src.speech_to_speech_llm import SpeechToSpeechLLM
from src.gorsellestirici import SpeechLLMGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 171 (FAZ 9): END-TO-END SPEECH-TO-SPEECH LLM: DUALLM / MOSHI DUAL-STREAM ARCHITECTURE")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. MODEL VE ÇİFT AKIŞLI GİRDİ HAZIRLIĞI
    # -------------------------------------------------------------
    print("\n[1/3] Speech-to-Speech Çift Başlıklı LLM Modeli Başlatılıyor...")
    model = SpeechToSpeechLLM(
        text_vocab_size=1000,
        audio_codebook_size=1024,
        num_quantizers=8,
        d_model=256,
        num_layers=4,
        num_heads=4,
    )

    # 2 saniyelik ses tokenları (25 Hz = 50 zaman adımı)
    audio_tokens_in = torch.randint(0, 1024, (1, 8, 50))
    text_tokens_in = torch.randint(0, 1000, (1, 50))

    with torch.no_grad():
        text_logits, audio_logits = model(audio_tokens_in, text_tokens_in)

    print(f"  • Kullanıcı Ses Token Girdisi  : {list(audio_tokens_in.shape)} [Batch=1, Num_Q=8, T=50]")
    print(f"  • Üretilen Metin Logitleri     : {list(text_logits.shape)} [Batch=1, T=50, Text_Vocab=1000]")
    print(f"  • Üretilen 8-Katman Ses Logiti : {list(audio_logits.shape)} [Batch=1, Num_Q=8, T=50, Codebook=1024]")

    # -------------------------------------------------------------
    # 2. CANLI SESLİ SOHBET VE GECİKME DEĞERLENDİRMESİ
    # -------------------------------------------------------------
    print("\n[2/3] Uçtan Uca Canlı Sesli Sohbet ve Gecikme Analizi Yapılıyor...")
    rapor = SpeechToSpeechLLM.ornek_diyalog_senaryolarini_getir()

    for s in rapor["senaryolar"]:
        print(f"\n>> DİYALOG: {s['diyalog_id']}")
        print(f"   Kullanıcı : \"{s['kullanici_sesi']}\"")
        print(f"   DuaLLM    : \"{s['asistan_yaniti_ses']}\"")
        print(f"   Gecikme   : {s['duallm_gecikme_ms']} ms (Geleneksel Zincir: {s['geleneksel_gecikme_ms']} ms)")

    print("\n" + "-" * 80)
    print(f"{'Performans Metriği':<40} | {'Değer'}")
    print("-" * 80)
    print(f"{'Geleneksel Boru Hattı Gecikmesi (Ortalama)':<40} | {rapor['ortalama_geleneksel_gecikme_ms']} ms")
    print(f"{'DuaLLM Uçtan Uca Gecikmesi (Ortalama)':<40} | {rapor['ortalama_duallm_gecikme_ms']} ms")
    print(f"{'Gecikme İyileşmesi (Hızlanma Faktörü)':<40} | {rapor['gecikme_iyilesmesi_kat']}x Daha Hızlı")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Speech-to-Speech LLM Teşhis Panosu Üretiliyor...")
    gorsellestirici = SpeechLLMGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "speech_to_speech_llm_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 171: SPEECH-TO-SPEECH LLM BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
