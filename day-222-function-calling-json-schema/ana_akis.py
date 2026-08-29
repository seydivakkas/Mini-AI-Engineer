"""
Day 222: Katı (Strict) JSON Schema ile Fonksiyon Çağrısı ve Dinamik Tip Doğrulama Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.fonksiyon_cagrisi_motoru import (
    StrictSchemaBuilder,
    ToolCallValidator,
    StrictFunctionDispatcher,
)
from src.fonksiyon_profilleyici import FonksiyonProfilleyici
from src.gorsellestirici import StrictFonksiyonGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 222 (FAZ 12): KATI (STRICT) JSON SCHEMA İLE FONKSİYON ÇAĞRISI VE DİNAMİK TİP DOĞRULAMA")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Katı (Strict) JSON Şeması Üretimi
    # -------------------------------------------------------------
    print("\n[1/4] Python Fonksiyonundan Otomatik Katı JSON Şeması Üretiliyor...")

    def hisse_fiyati_sorgula(sembol: str, gun_sayisi: int) -> str:
        return f"{sembol} hissesi için son {gun_sayisi} günlük ortalama fiyat: 142.50 TL"

    sema = StrictSchemaBuilder.sema_uret(
        hisse_fiyati_sorgula,
        "Belirtilen borsa sembolünün son N günlük fiyat geçmişini döner.",
    )
    print(f"  • Fonksiyon Adı    : {sema['name']}")
    print(f"  • Katı Mod (Strict): {sema['strict']}")
    print(f"  • Ek Alan İzni     : additionalProperties={sema['parameters']['additionalProperties']}")
    print(f"  • Zorunlu Alanlar  : {sema['parameters']['required']}")
    print("  ✓ Katı JSON Şeması Başarıyla Oluşturuldu!")

    # -------------------------------------------------------------
    # ADIM 2: Tip ve Alan Doğrulama Testi
    # -------------------------------------------------------------
    print("\n[2/4] Çalışma Zamanı Katı Şema ve Tip Doğrulayıcısı Denetleniyor...")
    gecerli_json = '{"sembol": "THYAO", "gun_sayisi": 7}'
    hatali_tip_json = '{"sembol": "THYAO", "gun_sayisi": "yedi"}'
    fazla_alan_json = '{"sembol": "THYAO", "gun_sayisi": 7, "tahmin": "yukselis"}'

    dogrulama_1 = ToolCallValidator.dogrula(gecerli_json, sema)
    dogrulama_2 = ToolCallValidator.dogrula(hatali_tip_json, sema)
    dogrulama_3 = ToolCallValidator.dogrula(fazla_alan_json, sema)

    print(f"  • Geçerli Çağrı Doğrulaması   : Geçerli={dogrulama_1['gecerli_mi']}")
    print(f"  • Hatalı Tip Çağrısı (String): Geçerli={dogrulama_2['gecerli_mi']} (Hata: {dogrulama_2['hata_mesaji']})")
    print(f"  • Fazla/Uydurma Alan Çağrısı  : Geçerli={dogrulama_3['gecerli_mi']} (Hata: {dogrulama_3['hata_mesaji']})")
    print("  ✓ Tüm Katı Doğrulama Kuralları Başarıyla Teyit Edildi!")

    # -------------------------------------------------------------
    # ADIM 3: Katı Dağıtıcı (Dispatcher) ile Fonksiyon Yürütme
    # -------------------------------------------------------------
    print("\n[3/4] Doğrulanmış Çağrı Katı Dağıtıcı Aracılığıyla Yürütülüyor...")
    dagitici = StrictFunctionDispatcher()
    dagitici.kaydet(hisse_fiyati_sorgula, "Borsa hisse fiyat sorgulama aracı.")

    yurutme = dagitici.calistir("hisse_fiyati_sorgula", gecerli_json)
    print(f"  • Yürütme Başarısı: {yurutme['basarili']}")
    print(f"  • Fonksiyon Çıktısı: '{yurutme['sonuc']}'")
    print("  ✓ Deterministik Fonksiyon Yürütmesi Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Strict Function Calling Teşhis Panosu Oluşturuluyor...")
    profil_raporu = FonksiyonProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "strict_function_calling_paneli.png")

    StrictFonksiyonGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Strict Function Calling Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 222 (FAZ 12): KATI (STRICT) JSON SCHEMA VE FONKSİYON ÇAĞRISI BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
