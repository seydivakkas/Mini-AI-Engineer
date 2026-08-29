"""
Day 235: SQL ve Veritabanı Analisti Ajanı (Text-to-SQL) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sql_ajani_motoru import (
    DatabaseSchema,
    SQLQueryReport,
    AgenticSQLAnalyst,
)
from src.sql_profilleyici import SQLProfilleyici
from src.gorsellestirici import SQLGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 235 (FAZ 12): SQL VE VERİTABANI ANALİSTİ AJAN (TEXT-TO-SQL) - ŞEMA BAĞLAMA VE OTONOM ONARIM")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Veritabanı Şeması ve Ajan Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] SQLite Bellek İçi Veritabanı ve Şema Yükleniyor...")
    analyst = AgenticSQLAnalyst()
    print(analyst.sema.sema_metni())

    # -------------------------------------------------------------
    # ADIM 2: Doğal Dil Sorusu ve Kendi Hatasını Düzelten SQL İcrası
    # -------------------------------------------------------------
    soru = "2026 yılında en çok harcama yapan müşterilerin isimleri ve toplam tutarları"
    print(f"\n[2/4] Kullanıcı Sorusu: '{soru}'")

    # 1. Hatalı Taslak (Var olmayan sütun: 'customer_name')
    hatali_sql = (
        "SELECT m.customer_name, SUM(s.tutar) "
        "FROM musteriler m "
        "JOIN siparisler s ON m.musteri_id = s.musteri_id "
        "GROUP BY m.customer_name"
    )

    # 2. Düzeltilmiş SQL ('ad_soyad')
    dogru_sql = (
        "SELECT m.ad_soyad, SUM(s.tutar) AS toplam_harcama "
        "FROM musteriler m "
        "JOIN siparisler s ON m.musteri_id = s.musteri_id "
        "WHERE s.tarih LIKE '2026%' "
        "GROUP BY m.ad_soyad "
        "ORDER BY toplam_harcama DESC"
    )

    rapor = analyst.sorgula_ve_analiz_et(soru, [hatali_sql, dogru_sql])

    print(f"\n  • Sorgu Başarılı mı?     : {rapor.basarili_mi}")
    print(f"  • Otonom Düzeltme Adımı  : {rapor.otonom_duzeltme_sayisi} (İlk denemedeki hata SQLite'da yakalanıp düzeltildi)")
    print(f"  • Nihai Doğru SQL        :\n    {rapor.sql_sorgusu}")

    # -------------------------------------------------------------
    # ADIM 3: Sonuç Tablosu ve Yönetici İçgörüsü
    # -------------------------------------------------------------
    print("\n[3/4] Dönen Tablo Kayıtları ve Doğal Dil Özeti:")
    print(f"  Sütunlar: {rapor.sutun_adlari}")
    for satir in rapor.donen_satirlar:
        print(f"    - {satir[0]}: {satir[1]:,.2f} TL")

    print(f"\n  💡 Doğal Dil Yönetici İçgörüsü:\n  {rapor.dogal_dil_icgorusu}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli SQL Analisti Teşhis Panosu Oluşturuluyor...")
    profil_raporu = SQLProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "sql_ajani_paneli.png")

    SQLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ SQL Analisti Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 235 (FAZ 12): SQL VE VERİTABANI ANALİSTİ AJAN BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
