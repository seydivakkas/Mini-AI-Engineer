"""
Day 65: SQLite Destekli CRUD, Model Çıkarım Logları ve AI Yönetim Paneli Ana Yürütme Betiği.
"""

import os
import sys
import pandas as pd

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.veritabani_yoneticisi import AIVeritabaniYoneticisi
from src.analiz_motoru import AITelemetriAnalizci
from src.gorsellestirici import DashboardGorsellestirici


def main():
    print("=" * 95, flush=True)
    print(">>> DAY 65: SQLITE DESTEKLI CRUD, MODEL CIKARIM LOGLARI & AI YONETIM PANELI", flush=True)
    print("=" * 95, flush=True)

    db_path = os.path.join(MEVCUT_DIZIN, "ai_yonetim.db")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    # 1. Veritabanı ve Tabloların Başlatılması
    print("\n[+] 1. Adim: SQLite WAL Modunda AI Yonetim Veritabani Baslatiliyor...", flush=True)
    db = AIVeritabaniYoneticisi(db_yolu=db_path)
    print(f"    - Veritabani Yolu  : {os.path.abspath(db_path)}")
    print(f"    - Tablolar         : cikarim_loglari, nesne_tespitleri (B-Tree Indeksli)")

    # 2. Sentetik Çıkarım Verisi Doldurma
    kayit_sayisi = 300
    print(f"\n[+] 2. Adim: {kayit_sayisi} Adet Sentetik Cikarim ve Nesne Tespiti Loglaniyor...", flush=True)
    AITelemetriAnalizci.sentetik_veri_doldur(db, kayit_sayisi=kayit_sayisi)

    # 3. Örnek CRUD İşlemleri (Create, Read, Update, Delete)
    print("\n[+] 3. Adim: SQLite CRUD Islemleri ve Insan Denetimi (Human-in-the-Loop) Kosuluyor...", flush=True)
    
    # Read with filter
    df_yuksek_guven = db.cikarimlari_getir(limit=10, min_guven=0.85)
    print(f"    - Yuksek Guvenli (>=0.85) Ilk 10 Kayit Cekildi (Boyut: {df_yuksek_guven.shape})")

    # Update: Human verification
    db.geri_bildirim_guncelle("trace_00005", dogru_mu=True, aciklama="Kusursuz tespit onaylandi")
    db.geri_bildirim_guncelle("trace_00010", dogru_mu=True, aciklama="Dogru tespit")
    db.geri_bildirim_guncelle("trace_00015", dogru_mu=False, aciklama="Yanlis pozitif etiketlendi")
    print(f"    - Insan Denetim Geri Bildirimleri Guncellendi (3 Kayit Isaretlendi)")

    # Delete
    silindi = db.cikarim_sil("trace_00099")
    print(f"    - Kayit Silme Testi (trace_00099) : {'Basarili' if silindi else 'Bulunamadi'}")

    # 4. Genel İstatistiklerin Hesaplanması
    stats = db.genel_istatistikleri_al()
    df_siniflar = AITelemetriAnalizci.sinif_dagilimi_al(db)
    df_tum_loglar = db.cikarimlari_getir(limit=300)

    print("\n" + "=" * 95, flush=True)
    print(">>> 4. AI CIKARIM TELEMETRI VE VERITABANI METRIKLERI", flush=True)
    print("=" * 95, flush=True)
    print(f"* Toplam Loglanan Istek     : {stats['toplam_istek']:,} Kayit")
    print(f"* Toplam Tespit Edilen Nesne: {stats['toplam_tespit']:,} Adet")
    print(f"* Ortalama Model Gecikmesi  : {stats['ortalama_gecikme_ms']:>6.2f} ms")
    print(f"* Ortalama Guven Skoru      : %{stats['ortalama_guven']*100:>5.1f}")
    print(f"* Insan Denetim Sayisi      : {stats['dogrulanan_adet']} Dogru / {stats['yanlis_adet']} Hatali")

    # 5. 6 Panelli Teşhis Panosunun Çizilmesi
    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "streamlit_sqlite_paneli.png")
    cikis = DashboardGorsellestirici.panel_ciz(
        istatistikler=stats,
        df_loglar=df_tum_loglar,
        df_siniflar=df_siniflar,
        hedef_path=hedef_pano
    )
    print(f"\n[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis)}", flush=True)
    print("=" * 95, flush=True)
    print("DAY 65: STREAMLIT SQLITE AI DASHBOARD BASARIYLA TAMAMLANDI!", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
