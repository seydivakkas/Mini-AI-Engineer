"""
Day 130: Human-in-the-Loop (HITL) Kesinti ve Güvenlik Ana Akışı (FAZ 7 BÜYÜK FİNALİ).
Dinamik risk eskalasyonu, kritik eylemlerde yürütmeyi duraklatma (Interrupt), insan düzenleme/onay/red mekanizması.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.hitl_kesinti_motoru import HITLOrkestratoru
from src.gorsellestirici import HITLGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 130: Human-in-the-Loop (HITL) Autonomous Agent Safety & Interrupts (FAZ 7 GRAND FINALE)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    orkestrator = HITLOrkestratoru()

    # -------------------------------------------------------------
    # ADIM 1: Karmaşık Eylem Planını Kuyruğa Ekle
    # -------------------------------------------------------------
    print("\n[1/4] Ajan Tarafından Planlanan Eylemler Kuyruğa Alınıyor...")
    orkestrator.eylem_ekle("log_sorgula", {"servis": "auth", "limit": 100})
    orkestrator.eylem_ekle("para_transferi", {"alici": "IBAN_TR_9988", "tutar": 85000.0})
    orkestrator.eylem_ekle("veritabani_tablo_sil", {"tablo": "musteriler_canli"})
    orkestrator.eylem_ekle("rapor_olustur", {"tur": "guvenlik_ozeti"})

    # -------------------------------------------------------------
    # ADIM 2: Adım Adım Yürütme ve Kesinti (Interrupt) Yönetimi
    # -------------------------------------------------------------
    print("\n[2/4] Yürütme Başlıyor (Risk Bazlı Dinamik Eskalasyon)...")

    # 1. Faz: Düşük riskli ilk eylem geçer, Para Transferi'nde duraklar
    durum1 = orkestrator.adim_adim_calistir()
    print(f"  [⏸️] KESİNTİ 1: {durum1['kesinti_eylemi'].eylem_adi} | Risk: %{durum1['kesinti_eylemi'].risk_skoru*100:.0f}")
    print(f"  [👤] İnsan Denetçi: 'Tutar çok yüksek (85.000 TL). 12.000 TL olarak DÜZENLE ve ONAYLA.'")
    durum2 = orkestrator.insan_karari_isle("DUZENLE", yeni_parametreler={"alici": "IBAN_TR_9988", "tutar": 12000.0})

    # 2. Faz: Veritabanı Tablo Silme'de duraklar
    print(f"  [⏸️] KESİNTİ 2: {durum2['kesinti_eylemi'].eylem_adi} | Risk: %{durum2['kesinti_eylemi'].risk_skoru*100:.0f}")
    print(f"  [👤] İnsan Denetçi: 'Canlı tablo silinemez! Eylemi REDDET ve güvenli arşiv al.'")
    durum3 = orkestrator.insan_karari_isle("REDDET", red_gerekcesi="Canlı veritabanı silme işlemi şirket politikası gereği yasaktır.")

    print(f"\n  [✓] Nihai Durum: {durum3['durum']}")
    print(f"  [✓] Tamamlanan / Engellenen Toplam Eylem Sayısı: {durum3['tamamlanan_sayisi']}")

    # -------------------------------------------------------------
    # ADIM 3: Denetim İzi (Audit Trail) Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Güvenlik Denetim İzi (Audit Trail) Raporu:")
    print("=" * 95)
    print(f"{'EYLEM ID':<12} | {'EYLEM ADI':<24} | {'RİSK':<10} | {'KARAR':<16} | {'DURUM'}")
    print("-" * 95)
    for e in orkestrator.tamamlanan_eylemler:
        print(f"{e.eylem_id:<12} | {e.eylem_adi:<24} | %{e.risk_skoru*100:<9.0f} | {e.insan_karari or 'OTOMATIK':<16} | {'İCRA EDİLDİ' if e.yurutuldu_mu else 'ENGELLENDİ'}")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 4: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli HITL & FAZ 7 BÜYÜK FİNALİ Teşhis Panosu Çiziliyor...")
    karsilastirma = orkestrator.benchmark_karsilastir()

    gorsellestirici = HITLGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "hitl_agent_paneli.png")
    gorsellestirici.pano_olustur(
        denetim_izi=orkestrator.denetim_izi,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 130: HUMAN-IN-THE-LOOP & FAZ 7 BÜYÜK FİNALİ (CAPSTONE) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
