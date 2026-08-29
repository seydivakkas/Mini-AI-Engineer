"""
Day 256: Ses Komutlu Robot Ajanı (Whisper + VLM + VLA) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.voice_robot_motoru import (
    WhisperSemanticParser,
    VisualSpatialGrounder,
    VoiceConditionedVLAAgent,
)
from src.voice_robot_profilleyici import VoiceRobotProfilleyici
from src.gorsellestirici import VoiceRobotGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 256 (FAZ 13): SES KOMUTLU ROBOT AJANI (WHISPER + VLM + VLA İLE UÇTAN UCA ROBOT İDARESİ)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Sesli Robot Ajanının Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Whisper ASR, VLM Semantik Temellendirici ve VLA İcra Ajanı Başlatılıyor...")
    agent = VoiceConditionedVLAAgent()
    print(f"  • Robot Durumu                : {agent.durum}")
    print(f"  • Güvenlik Hız Limiti (Maks)  : {agent.maks_hiz_m_s} m/s")

    # -------------------------------------------------------------
    # ADIM 2: Doğal Ses Komutu İcrası
    # -------------------------------------------------------------
    ses_komutu_1 = "Lütfen masadaki kırmızı kupayı alıp su ısıtıcısının yanına koyar mısın?"
    print(f"\n[2/4] Ses Komutu Alındı: \"{ses_komutu_1}\"")
    sonuc_1 = agent.process_voice_instruction(ses_komutu_1)

    print(f"  • Çözümlenen Niyet             : {sonuc_1['parse_sonucu']['niyet']}")
    print(f"  • Hedef Nesne (3D Temellendirme): {sonuc_1['nesne_grounding']['nesne_adi']} -> {sonuc_1['nesne_grounding']['koordinat_3d_m']}")
    print(f"  • Hedef Konum (3D Temellendirme): {sonuc_1['hedef_grounding']['nesne_adi']} -> {sonuc_1['hedef_grounding']['koordinat_3d_m']}")
    print(f"  • VLA Alt Görev Sayısı         : {len(sonuc_1['icra_plani'])} Adım")
    print(f"  • Robotun Sesli Geri Bildirimi : \"{sonuc_1['sesli_geri_bildirim']}\"")

    # -------------------------------------------------------------
    # ADIM 3: Belirsizlik ve Netleştirme Senaryosu
    # -------------------------------------------------------------
    ses_komutu_2 = "Şuradaki bardağı alıp masaya koy"
    print(f"\n[3/4] Belirsiz Ses Komutu Alındı: \"{ses_komutu_2}\"")
    sonuc_2 = agent.process_voice_instruction(ses_komutu_2)

    print(f"  • Belirsizlik Durumu           : {sonuc_2['parse_sonucu']['belirsizlik_var_mi']}")
    print(f"  • Robotun Netleştirme Sorusu   : \"{sonuc_2['sesli_geri_bildirim']}\"")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Ses Komutlu Robot Teşhis Panosu Oluşturuluyor...")
    profil_raporu = VoiceRobotProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "voice_robot_paneli.png")

    VoiceRobotGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Sesli Robot Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 256 (FAZ 13): SES KOMUTLU ROBOT AJANI MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
