"""
Day 121: ReAct (Reasoning + Acting) Otonom AI Ajanı & Scratchpad Bellek Ana Akışı (Faz 7 Başlangıcı).
Thought-Action-Observation döngüsü, araç entegrasyonu ve çok adımlı problem çözme gösterimi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.araclar import (
    AracKayitDefteri,
    HesapMakinasi,
    AramaMotoru,
    PythonCalistirici,
)
from src.react_ajan import ReActAjani
from src.gorsellestirici import ReActGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 121: FAZ 7 BAŞLANGICI - ReAct (Reasoning + Acting) AI Agent & Scratchpad Architecture")
    print("=" * 105)

    # -------------------------------------------------------------
    # ADIM 1: Araç Seti ve Kayıt Defteri Kurulumu
    # -------------------------------------------------------------
    print("\n[1/3] Güvenli Araç Kayıt Defteri (Tool Registry) Başlatılıyor...")
    kayit = AracKayitDefteri()
    kayit.arac_ekle(HesapMakinasi())
    kayit.arac_ekle(AramaMotoru())
    kayit.arac_ekle(PythonCalistirici())

    print(kayit.sistem_promptu_aciklamasi())

    # -------------------------------------------------------------
    # ADIM 2: ReAct Ajanının Çalıştırılması ve Trajectory İzleme
    # -------------------------------------------------------------
    print("\n[2/3] ReAct Otonom Ajanı Görev Üzerinde Koşturuluyor...")
    ajan = ReActAjani(arac_kayit=kayit, maksimum_iterasyon=6, seed=42)

    gorev = "Türkiye'nin başkenti neresidir ve 2024 nüfusu kaçtır?"
    print(f"\n[-] HEDEF GÖREV: '{gorev}'\n")

    sonuc = ajan.calistir(gorev)

    print("=" * 70)
    print("                🧠 REACT SCRATCHPAD HAFIZA TRAJECTORY                ")
    print("=" * 70)
    print(sonuc["scratchpad"])
    print("=" * 70)
    print(f"\n[✓] NİHAİ YANIT : {sonuc['nihai_yanit']}")
    print(f"[✓] DURUM       : {'BAŞARILI' if sonuc['basarili'] else 'BAŞARISIZ'} ({sonuc['toplam_adim']} Adımda Tamamlandı)")

    # -------------------------------------------------------------
    # ADIM 3: CoT vs Act-Only vs ReAct Mimari Karşılaştırması
    # -------------------------------------------------------------
    print("\n[3/3] Mimari Kıyaslama (CoT vs Act-Only vs ReAct) ve Teşhis Panosu Çiziliyor...")
    karsilastirma = ajan.mimari_karsilastir()

    print("\n" + "=" * 90)
    print(f"{'MİMARİ':<28} | {'DOĞRULUK (%)':<14} | {'HALÜSİNASYON (%)':<18} | {'HATA KURTARMA (%)':<18}")
    print("-" * 90)
    for m, d, h, hk in zip(
        karsilastirma["modeller"],
        karsilastirma["dogruluk_orani"],
        karsilastirma["halusinasyon_orani"],
        karsilastirma["hata_kurtarma_orani"],
    ):
        print(f"{m:<28} | %{d:>11.1f} | %{h:>15.1f} | %{hk:>15.1f}")
    print("-" * 90)

    gorsellestirici = ReActGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "react_ajan_paneli.png",
    )
    gorsellestirici.pano_olustur(
        calisma_sonucu=sonuc,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 121: ReAct AJANI VE SCRATCHPAD MİMARİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
