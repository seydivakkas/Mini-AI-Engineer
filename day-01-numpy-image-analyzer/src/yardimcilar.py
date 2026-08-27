"""NumPy Görüntü Analizörü - Yardımcı Araçlar ve Sentetik Veri Üreticileri.

Bu modül, harici görsel kütüphanelerine (OpenCV, PIL vb.) ihtiyaç duymadan
doğrudan matematiksel fonksiyonlar ve NumPy vektörizasyonu ile test görselleri
üretmeyi ve bellek/adım (stride) özelliklerini incelemeyi sağlar.
"""

from typing import Any, Dict
import numpy as np


def sentetik_goruntu_uret(
    yukseklik: int = 256,
    genislik: int = 256,
    desen_tipi: str = "gradyan"
) -> np.ndarray:
    """Belirtilen desen tipinde ve boyutlarda sentetik RGB piksel matrisi üretir.

    Parametreler:
        yukseklik (int): Üretilecek görselin piksel yüksekliği (Y ekseni).
        genislik (int): Üretilecek görselin piksel genişliği (X ekseni).
        desen_tipi (str): 'gradyan', 'dama_tahtasi' veya 'renkli_bloklar'.

    Döndürür:
        np.ndarray: uint8 tipinde, (yukseklik, genislik, 3) boyutunda RGB matrisi.

    Hatalar:
        ValueError: Desteklenmeyen desen tipi veya geçersiz boyut girildiğinde.
    """
    if yukseklik <= 0 or genislik <= 0:
        raise ValueError(
            f"Görsel boyutları pozitif tam sayı olmalıdır. Girilen: ({yukseklik}, {genislik})"
        )

    if desen_tipi == "gradyan":
        # X ve Y ekseni boyunca doğrusal gradyanlar oluşturulur
        x_vektoru = np.linspace(0, 255, genislik, dtype=np.float32)
        y_vektoru = np.linspace(0, 255, yukseklik, dtype=np.float32)
        x_izgarasi, y_izgarasi = np.meshgrid(x_vektoru, y_vektoru)

        # Kırmızı: X gradyanı, Yeşil: Y gradyanı, Mavi: İkisinin ortalaması
        kirmizi = x_izgarasi.astype(np.uint8)
        yesil = y_izgarasi.astype(np.uint8)
        mavi = ((x_izgarasi + y_izgarasi) / 2.0).astype(np.uint8)

        return np.stack([kirmizi, yesil, mavi], axis=-1)

    elif desen_tipi == "dama_tahtasi":
        kare_boyutu = max(8, min(yukseklik, genislik) // 8)
        y_indeksleri = np.arange(yukseklik) // kare_boyutu
        x_indeksleri = np.arange(genislik) // kare_boyutu
        x_izgara, y_izgara = np.meshgrid(x_indeksleri, y_indeksleri)

        desen = ((x_izgara + y_izgara) % 2).astype(np.uint8) * 255
        # 3 kanala da aynı deseni yayarak siyah-beyaz RGB elde edilir
        return np.stack([desen, desen, desen], axis=-1)

    elif desen_tipi == "renkli_bloklar":
        matris = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
        orta_y = yukseklik // 2
        orta_x = genislik // 2

        # Sol Üst: Kırmızı blok
        matris[:orta_y, :orta_x, 0] = 255
        # Sağ Üst: Yeşil blok
        matris[:orta_y, orta_x:, 1] = 255
        # Sol Alt: Mavi blok
        matris[orta_y:, :orta_x, 2] = 255
        # Sağ Alt: Sarı blok (Kırmızı + Yeşil)
        matris[orta_y:, orta_x:, 0] = 255
        matris[orta_y:, orta_x:, 1] = 255

        return matris

    else:
        desteklenenler = ["gradyan", "dama_tahtasi", "renkli_bloklar"]
        raise ValueError(
            f"Geçersiz desen tipi: '{desen_tipi}'. Desteklenen tipler: {desteklenenler}"
        )


def bellek_ve_stride_raporla(dizi: np.ndarray) -> Dict[str, Any]:
    """Verilen NumPy dizisinin bellek yerleşimi ve adım (stride) metaverilerini çıkarır.

    Parametreler:
        dizi (np.ndarray): İncelenecek piksel veya öznitelik dizisi.

    Döndürür:
        Dict[str, Any]: Boyut, veri tipi, bellek baytı, strides ve bellek bayrakları sözlüğü.
    """
    if not isinstance(dizi, np.ndarray):
        raise TypeError(f"Girdi bir numpy dizisi olmalıdır, alınan: {type(dizi)}")

    return {
        "boyut_sekli": dizi.shape,
        "veri_tipi": str(dizi.dtype),
        "eleman_sayisi": int(dizi.size),
        "toplam_bellek_bayt": int(dizi.nbytes),
        "eleman_basi_bayt": int(dizi.itemsize),
        "adimlar_strides": dizi.strides,
        "c_surekli_mi": bool(dizi.flags.c_contiguous),
        "fortran_surekli_mi": bool(dizi.flags.f_contiguous),
        "kendi_verisine_sahip_mi": bool(dizi.flags.owndata),
    }
