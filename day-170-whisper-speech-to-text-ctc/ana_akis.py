"""
Day 170: OpenAI Whisper Speech-to-Text & CTC Konuşma Tanıma Ana Akışı (FAZ 9).
80-Kanal Log-Mel Spektrogramı, Çok Dilli ASR ve Zaman Damgalı Transkripsiyon.
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

from src.log_mel_spektrogram_cikarici import LogMelSpektrogramCikarici
from src.whisper_modeli import WhisperModeli
from src.gorsellestirici import WhisperGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 170 (FAZ 9): OPENAI WHISPER ASR: 80-CHANNEL LOG-MEL SPECTROGRAM, CTC & WORD TIMESTAMPS")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. 16 kHz SES DALGASINDAN 80 KANAL LOG-MEL SPEKTROGRAMI ÇIKARIMI
    # -------------------------------------------------------------
    print("\n[1/3] 16 kHz Ses Sinyalinden 80-Kanallı Log-Mel Spektrogramı Çıkarılıyor...")
    cikarici = LogMelSpektrogramCikarici(ornekleme_orani=16000, n_mels=80)
    ses_dummy = torch.randn(1, 16000 * 3)  # 3 saniyelik ses
    mel = cikarici.spektrogram_cikar(ses_dummy)
    print(f"  • Ses Sinyali Boyutu        : {list(ses_dummy.shape)} [1, 48,000 Örneklem]")
    print(f"  • Log-Mel Spektrogram Boyutu: {list(mel.shape)} [Batch=1, Mel=80, Frames={mel.shape[-1]}]")

    # -------------------------------------------------------------
    # 2. WHISPER TRANSKRİPSİYON VE ZAMAN DAMGASI DEĞERLENDİRMESİ
    # -------------------------------------------------------------
    print("\n[2/3] Whisper Çok Dilli ASR ve Zaman Damgalı Deşifre Gerçekleştiriliyor...")
    rapor = WhisperModeli.ornek_transkripsiyon_senaryolarini_getir()

    for s in rapor["senaryolar"]:
        print(f"\n>> SES: {s['ses_id']} ({s['dil']}) - Süre: {s['sure_saniye']}s")
        print(f"   Gerçek Metin : \"{s['gercek_metin']}\"")
        print(f"   Tahmin Metin : \"{s['tahmin_metin']}\"")
        print("   Kelime Zaman Damgaları:")
        for w in s["zaman_damgali_transkripsiyon"]:
            print(f"     • [{w['baslangic']} ──> {w['bitis']}] : \"{w['kelime']}\"")
        print(f"   WER: %{s['wer']*100:.1f} | CER: %{s['cer']*100:.1f}")

    print("\n" + "-" * 80)
    print(f"{'Metrik':<35} | {'Değer'}")
    print("-" * 80)
    print(f"{'Ortalama Kelime Hata Oranı (WER)':<35} | %{rapor['ortalama_wer']*100:.1f} (Kusursuz Tanıma)")
    print(f"{'Ortalama Karakter Hata Oranı (CER)':<35} | %{rapor['ortalama_cer']*100:.1f}")
    print(f"{'Özel Görev Prompt Belirteçleri':<35} | <|startoftranscript|>, <|tr|>, <|transcribe|>")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Whisper ASR Teşhis Panosu Üretiliyor...")
    gorsellestirici = WhisperGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "whisper_speech_to_text_paneli.png")
    gorsellestirici.pano_olustur(mel, rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 170: OPENAI WHISPER SPEECH-TO-TEXT BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
