"""
Day 232: Human-in-the-Loop (HITL) Güvenlik Bariyeri Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.hitl_motoru import (
    RiskLevel,
    ActionRequest,
    ApprovalDecision,
    HITLGuardrailAgent,
)
from src.hitl_profilleyici import HITLProfilleyici
from src.gorsellestirici import HITLGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 232 (FAZ 12): HUMAN-IN-THE-LOOP (HITL) GÜVENLİK BARİYERİ - KRİTİK İŞLEMLERDE İNSAN ONAYI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: HITL Güvenlik Bariyeri Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] HITL Güvenlik Bariyeri Başlatılıyor...")
    ajan = HITLGuardrailAgent()
    print("  ✓ Risk Derecelendirici ve Interrupt Kapısı Aktif.")

    # -------------------------------------------------------------
    # ADIM 2: Düşük Riskli Eylem (Otomatik Bypass)
    # -------------------------------------------------------------
    print("\n[2/4] Düşük Riskli Veritabanı Okuma Talebi Değerlendiriliyor...")
    talep_okuma = ajan.eylem_talebi_olustur(
        arac_adi="query_database",
        parametreler={"query": "SELECT COUNT(*) FROM siparisler"},
        gerekce="Günlük toplam sipariş sayısını öğrenmek.",
    )
    sonuc_okuma = ajan.eylemi_denetle_ve_icra_et(talep_okuma)
    print(f"  • Risk Seviyesi : {sonuc_okuma['risk']}")
    print(f"  • Durum         : {sonuc_okuma['durum']}")
    print(f"  • Çıktı         : {sonuc_okuma['mesaj']}")

    # -------------------------------------------------------------
    # ADIM 3: Kritik Riskli Eylem ve İnsan Onay Akışı (Interrupt)
    # -------------------------------------------------------------
    print("\n[3/4] Kritik Riskli Tablo Silme Talebi (HITL Interrupt)...")
    talep_silme = ajan.eylem_talebi_olustur(
        arac_adi="delete_database_table",
        parametreler={"table": "musteri_veritabanı_prod"},
        gerekce="Disk alanını boşaltmak için eski tabloları silmek.",
    )

    # Aşama 3a: Onaysız çalıştırma girişimi (Dondurma)
    sonuc_dondurma = ajan.eylemi_denetle_ve_icra_et(talep_silme)
    print(f"  • [Aşama 3a] : {sonuc_dondurma['mesaj']}")

    # Aşama 3b: İnsan inceleme ve Red kararı
    print("\n  [İNSAN İNCELEME PANELİ]: Mühendis inceledi ve red gerekçesi girdi:")
    insan_reddi = ApprovalDecision(
        onaylandi_mi=False,
        insan_yorumu="Canlı müşteri tablosu silinemez! Bunun yerine verileri S3'e arşivle.",
    )
    sonuc_nihai = ajan.eylemi_denetle_ve_icra_et(talep_silme, insan_reddi)
    print(f"  • [Aşama 3b] : {sonuc_nihai['mesaj']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli HITL Guardrail Teşhis Panosu Oluşturuluyor...")
    profil_raporu = HITLProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "hitl_guardrail_paneli.png")

    HITLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ HITL Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 232 (FAZ 12): HUMAN-IN-THE-LOOP GÜVENLİK BARİYERİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
