"""
Day 236: Hiyerarşik Görev Delegasyonu Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.hiyerarsi_ajani_motoru import (
    SubTask,
    WorkerAgent,
    ManagerAgent,
)
from src.hiyerarsi_profilleyici import HiyerarsiProfilleyici
from src.gorsellestirici import HiyerarsiGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 236 (FAZ 12): HİYERARŞİK GÖREV DELEGASYONU - YÖNETİCİ VE İŞÇİ AJANLAR ARASINDA YÜK PAYLAŞIMI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Yönetici ve Uzman İşçi Ajanların Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] Yönetici Ajan (Manager) ve Alan İşçileri (Workers) Kaydediliyor...")
    manager = ManagerAgent()
    manager.isci_kaydet(WorkerAgent("database", "PostgreSQL & Migration Uzmanı"))
    manager.isci_kaydet(WorkerAgent("backend", "FastAPI & JWT Güvenlik Uzmanı"))
    manager.isci_kaydet(WorkerAgent("security", "Siber Güvenlik & Penetrasyon Uzmanı"))

    for alan, isci in manager.isciler.items():
        print(f"  • Uzman İşçi: [{alan.upper()}] -> {isci.uzmanlik}")

    # -------------------------------------------------------------
    # ADIM 2: Kök Hedef ve İş Kırılım Yapısı (WBS)
    # -------------------------------------------------------------
    kok_hedef = "Kimlik Doğrulama Mikroservisini Kur ve Güvenliğini Doğrula"
    print(f"\n[2/4] Kök Hedef Alındı: '{kok_hedef}'")

    alt_gorevler = manager.gorevi_ayristir(kok_hedef)
    print("\n  📋 İş Kırılım Yapısı (WBS / Subtasks):")
    for t in alt_gorevler:
        print(f"    - Görev #{t.task_id} [{t.alan.upper()}]: {t.aciklama}")

    # -------------------------------------------------------------
    # ADIM 3: Delegasyon İcrası ve Sentez
    # -------------------------------------------------------------
    print("\n[3/4] Görevler Uzman İşçilere Delege Ediliyor ve Sentezleniyor...")
    sonuc = manager.gorevleri_delege_et_ve_birlestir(kok_hedef)

    print("\n--- [Yönetici İcra Günlüğü] ---")
    for log in sonuc["gunluk"]:
        print("  " + log)

    print(f"\n  🎯 Nihai Sentez Raporu:\n  {sonuc['sentez']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Hiyerarşik Delegasyon Teşhis Panosu Oluşturuluyor...")
    profil_raporu = HiyerarsiProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "hiyerarsi_ajani_paneli.png")

    HiyerarsiGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Hiyerarşi Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 236 (FAZ 12): HİYERARŞİK GÖREV DELEGASYONU BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
