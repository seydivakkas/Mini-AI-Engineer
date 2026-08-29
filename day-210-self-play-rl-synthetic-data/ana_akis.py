"""
Day 210: Self-Play RL ve Sentetik Veri Döngüsü Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.self_play_motoru import (
    SyntheticProblemGenerator,
    ReasoningSolver,
    SelfPlayReferee,
    CurriculumScheduler,
    SelfPlayRLTrainer,
)
from src.self_play_profilleyici import SelfPlayProfilleyici
from src.gorsellestirici import SelfPlayGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 210 (FAZ 11): SELF-PLAY RL & SYNTHETIC DATA CURRICULUM ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Sentetik Problem Üretimi ve Seviye Testi
    # -------------------------------------------------------------
    print("\n[1/4] Dinamik Zorluk Kademelerinde Sentetik Problem Üretimi...")
    for z in [2.0, 5.5, 9.0]:
        p = SyntheticProblemGenerator.problem_uret(zorluk=z)
        print(f"  • [Zorluk δ={z:<3.1f}] Soru: '{p['soru']}' -> Doğru Cevap: {p['dogru_cevap']} ({p['kategori']})")
    print("  ✓ Sentetik Problem Üreticisi Başarıyla Test Edildi!")

    # -------------------------------------------------------------
    # ADIM 2: Akıl Yürüten Çözücü ve Hakem Değerlendirmesi
    # -------------------------------------------------------------
    print("\n[2/4] Problem Çözücü ve Deterministik Hakem Ödül Mekanizması...")
    solver = ReasoningSolver(yetenek_theta=3.0)
    test_prob = SyntheticProblemGenerator.problem_uret(zorluk=3.0)
    cozum = solver.cozumu_yurut(test_prob)
    odul = SelfPlayReferee.odul_hesapla(test_prob, cozum)

    print(f"  • Çözülen Soru    : {test_prob['soru']}")
    print(f"  • Üretilen Yanıt  : {cozum['uretilen_cevap']} (Doğru mu: {'✅ Evet' if odul['dogru_mu'] else '❌ Hayır'})")
    print(f"  • Çözücü Ödülü    : R_solver = {odul['r_solver']:.2f}")
    print(f"  • Üretici Ödülü   : R_generator = {odul['r_generator']:.2f} (Frontier Dengesi)")
    print("  ✓ Karşılıklı Hakem Ödülü Başarıyla Hesaplandı!")

    # -------------------------------------------------------------
    # ADIM 3: 100 Turluk Self-Play Simülasyonu
    # -------------------------------------------------------------
    print("\n[3/4] 100 Turluk Otonom Self-Play Kendi Kendine Öğrenme Döngüsü Yürütülüyor...")
    profil_raporu = SelfPlayProfilleyici.simulasyon_yurut(toplam_tur=100)

    print(f"  • Başlangıç Zorluğu (δ) : {profil_raporu['baslangic_zorluk']:.1f}")
    print(f"  • Bitiş Zorluğu (δ)     : {profil_raporu['son_zorluk']:.1f} (Zorluk +{profil_raporu['son_zorluk'] - profil_raporu['baslangic_zorluk']:.1f} Arttı)")
    print(f"  • Başlangıç Yeteneği(θ) : {profil_raporu['baslangic_yetenek']:.1f}")
    print(f"  • Bitiş Yeteneği (θ)    : {profil_raporu['son_yetenek']:.1f} (Model Otonom Olarak Güçlendi)")
    print(f"  • Genel Çözüm Başarısı  : %{profil_raporu['genel_dogruluk_orani']:.1f}")
    print("  ✓ Self-Play Müfredat Büyümesi Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Self-Play RL Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "self_play_rl_paneli.png")

    SelfPlayGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Self-Play Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 210 (FAZ 11): SELF-PLAY RL & SENTETİK VERİ DÖNGÜSÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
