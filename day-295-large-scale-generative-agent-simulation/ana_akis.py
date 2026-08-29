"""
Day 295 (FAZ 15): Büyük Ölçekli Üretken Ajan Simülasyonu Ana Akış Betiği.
Stanford Smallville Mimarisi, Bellek Akışı, Refleksiyon, Günlük Planlama ve Sosyal Yayılım.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.generative_agent_motoru import (
    EpisodicMemory,
    MemoryStreamRetriever,
    GenerativeAgent,
    SocialTownSimulation,
)
from src.generative_agent_profilleyici import GenerativeAgentProfilleyici
from src.gorsellestirici import GenerativeAgentGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 295 (FAZ 15): BÜYÜK ÖLÇEKLİ ÜRETKEN AJAN SİMÜLASYONU VE DİJİTAL TOPLUM — GENERATIVE AGENTS")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Kasaba ve Üretken Ajanların Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Stanford Smallville Dijital Kasaba ve Üretken Ajanlar Başlatılıyor...")
    agent = GenerativeAgent("Klaus", "Üniversite Öğrencisi")
    print(f"  • Ajan Adı ve Rolü                   : {agent.name} ({agent.role})")
    print(f"  • Günlük Plan Adım Sayısı            : {len(agent.daily_plan)} Etkinlik")
    print(f"  • Örnek Günlük Plan                  : {agent.daily_plan[0]} -> {agent.daily_plan[-1]}")

    # -------------------------------------------------------------
    # ADIM 2: Epizodik Bellek Akışı ve Refleksiyon Üretimi
    # -------------------------------------------------------------
    print("\n[2/4] Epizodik Bellek Akışına Olaylar Kaydediliyor ve Üst Düzey Refleksiyon Üretiliyor...")
    agent.add_memory("Maria ile kafede karşılaştım ve akşam 18:00'deki partiyi öğrendim.", timestamp=10, importance=0.92)
    agent.add_memory("Kütüphanede yapay zeka ve simülasyon makalesi okudum.", timestamp=12, importance=0.65)
    insight = agent.reflect()

    print(f"  • Kaydedilen Epizodik Anı Sayısı     : {len(agent.memory_stream)}")
    print(f"  • Sentezlenen Üst Düzey Refleksiyon  : \"{insight}\"")

    # -------------------------------------------------------------
    # ADIM 3: Organik Bilgi Yayılımı ve Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Kasaba Halkı Arasında Organik Bilgi Yayılımı (Diffusion) Simüle Ediliyor...")
    profil = GenerativeAgentProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Bilgi Yayılım Başarısı (4 Saat)    : %25.0 -> %{profil['yayilim_oranlari'][-1]:.1f}")
    print(f"  • İnsan Gerçekçilik & İnandırıcılık  : %{kars['inandiricilik_skoru_yuzde']['1. Static FSM NPC']:.1f} -> %{kars['inandiricilik_skoru_yuzde']['3. Generative Agent']:.1f} (+%{profil['gercekcilik_artisi']:.1f})")
    print(f"  • Uzun Vadeli Bellek Doğruluğu       : %{kars['bellek_erisim_dogrulugu_yuzde']['3. Generative Agent']:.1f}")
    print(f"  • 24 Saatlik Davranış Tutarlılığı    : %{kars['davranis_tutarliligi_yuzde']['3. Generative Agent']:.1f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Üretken Ajan Simülasyon Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "generative_agent_simulation_paneli.png")

    GenerativeAgentGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Üretken Ajan Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 295 (FAZ 15): BÜYÜK ÖLÇEKLİ ÜRETKEN AJAN SİMÜLASYONU MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
