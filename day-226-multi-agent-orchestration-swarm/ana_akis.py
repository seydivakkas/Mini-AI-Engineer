"""
Day 226: Çoklu Ajan Orkestrasyonu (Swarm) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.swarm_motoru import (
    AgentMessage,
    SpecializedAgent,
    ResearcherAgent,
    CoderAgent,
    ReviewerAgent,
    SwarmOrchestrator,
)
from src.swarm_profilleyici import SwarmProfilleyici
from src.gorsellestirici import SwarmGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 226 (FAZ 12): ÇOKLU AJAN ORKESTRASYONU (SWARM) - HİYERARŞİK İLETİŞİM VE İŞBİRLİĞİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Swarm Orkestratörü ve Uzman Ajanların Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] Swarm Orkestratörü Başlatılıyor ve Uzman Ajanlar Kaydediliyor...")
    orkestrator = SwarmOrchestrator()
    print("  ✓ Kayıtlı Ajanlar: ['Yönetici (Orchestrator)', 'Araştırmacı', 'Kodlayıcı', 'Denetçi (QA)']")

    # -------------------------------------------------------------
    # ADIM 2: Hiyerarşik Görev Dağıtımı ve İcra
    # -------------------------------------------------------------
    hedef = "Hızlı Sıralama (Quicksort) Algoritması Geliştirme ve Testi"
    print(f"\n[2/4] Çoklu Ajan İşbirliği Başlatılıyor (Hedef: '{hedef}')...")

    sonuc = orkestrator.gorev_dagit_ve_sentezle(ana_hedef=hedef)

    print("\n--- [Ajanlar Arası Mesaj Veriyolu / Message Bus] ---")
    for iz in sonuc["mesaj_izleri"]:
        print("  " + iz)

    print("\n--- [Nihai Sentez Raporu] ---")
    print(sonuc["nihai_cikti"])

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli Swarm Orkestrasyon Teşhis Panosu Oluşturuluyor...")
    profil_raporu = SwarmProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "swarm_orkestrasyon_paneli.png")

    SwarmGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Swarm Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 226 (FAZ 12): ÇOKLU AJAN ORKESTRASYONU (SWARM) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
