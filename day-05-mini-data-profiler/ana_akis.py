"""Günün Ana Çalıştırma Akışı: Otomatik Mini Veri Seti Profilleyici.

Bu betik; yapay zeka ve bilgisayarlı görü projelerinde kullanılan görsel metaveri
tablosunu otomatik olarak profiller; sütun kardinalitesi, çarpıklık, basıklık ve
kayıp veri oranlarını tespit ederek hem konsola hem de Markdown raporuna basar.
"""

import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekler
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
import pandas as pd
from src.veri_profilleyici import MiniVeriProfilleyici
from src.rapor_olusturucu import ProfilRaporOlusturucu


def main() -> None:
    # 1. Örnek Veri Seti Oluşturma (Farklı dağılım ve anomali kalıpları içeren)
    np.random.seed(42)
    ornek_sayisi = 1200

    veri = pd.DataFrame({
        # Yüksek kardinaliteli aday anahtar
        "gorsel_kodu": [f"IMG_{i:06d}" for i in range(ornek_sayisi)],
        # Simetrik sürekli sayısal
        "en_boy_orani": np.random.normal(loc=1.77, scale=0.08, size=ornek_sayisi),
        # Sağa aşırı çarpık (exponential) dosya boyutu
        "dosya_boyutu_mb": np.random.exponential(scale=2.5, size=ornek_sayisi) + 0.2,
        # Sıfır varyanslı sabit sütun
        "renk_derinligi_bit": [24] * ornek_sayisi,
        # Yüksek eksik veri oranlı (%25+) kategorik sütun
        "dokuma_tezgah_no": np.random.choice(
            ["Tezgah_A", "Tezgah_B", "Tezgah_C", None],
            p=[0.25, 0.25, 0.20, 0.30],
            size=ornek_sayisi
        ),
        # Ayrık ikili etiket
        "kalite_onay": np.random.choice([0, 1], p=[0.10, 0.90], size=ornek_sayisi)
    })

    # 2. Profilleme Motorunu Çalıştırma
    profilleyici = MiniVeriProfilleyici(
        yuksek_eksik_esigi=0.20,
        yuksek_carpiklik_esigi=1.5
    )
    profil = profilleyici.profili_cikar(veri)

    # 3. Konsola Detaylı Rapor Basma
    ProfilRaporOlusturucu.konsola_yazdir(profil)

    # 4. Markdown Raporu Kaydetme
    md_icerik = ProfilRaporOlusturucu.markdown_raporu_uret(profil)
    cikti_yolu = proje_kok / "profil_raporu.md"
    with open(cikti_yolu, "w", encoding="utf-8") as f:
        f.write(md_icerik)

    print(f"\n[V] Markdown raporu başarıyla üretildi: {cikti_yolu.name}")
    print("[V] Day 5: Mini Veri Profilleyici başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
