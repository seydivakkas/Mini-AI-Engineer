# Day 51: Pillow ile Hataya Toleranslı ve Güvenli Görsel Yükleyici (Safe & Fault-Tolerant Image Loader)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Pillow](https://img.shields.io/badge/Pillow-10.0+-blueviolet.svg?style=flat-square&logo=python)](https://python-pillow.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-013243.svg?style=flat-square&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB.svg?style=flat-square)](https://seaborn.pydata.org/)
[![Testler](https://img.shields.io/badge/pytest-8.0+-green.svg?style=flat-square)](https://docs.pytest.org/)
[![Lisans: Tüm Hakları Saklıdır](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](./LICENSE)

Bu proje; **FAZ 3: Çekirdek ML/DL Boru Hatları, Optimizasyon ve Edge MLOps** müfredatımızın 51. gününde geliştirilen **Üretim Seviyesi Hataya Toleranslı ve Güvenli Görsel Yükleme (Safe Image Ingestion Pipeline) Motorudur**. Canlı web servislerine ve veri kümelerine gelen görsellerdeki Decompression Bomb (bellek taşması/DoS) saldırılarını engeller, EXIF oryantasyon açılarını otomatik düzeltir, şeffaf RGBA/CMYK formatlarını standart RGB tensörlerine dönüştürür ve kesik/bozuk ağ akışlarını kurtarır.

---

## 📖 Mentorluk Dersi ve Görsel Yükleme Güvenliği

### 1. Neden Standart `Image.open()` Üretimde Tehlikelidir?

Gerçek dünya Bilgisayarlı Görü (Computer Vision) üretim ortamlarında (FastAPI görsel yükleme uçları, S3 kuyruk işleyicileri, PyTorch DataLoader), istemcilerden gelen ham görseller doğrudan sinir ağına beslenemez:

1. **Decompression Bomb Saldırısı (DoS & Memory Exhaustion):**
   - Sıkıştırılmış dosya boyutu yalnızca $50\text{ KB}$ olan bir PNG, açıldığında $50,000 \times 50,000$ piksel olabilir.
   - Açılmış tensör belleği:
     $$\text{RAM İhtiyacı} = 50,000 \times 50,000 \times 3 \text{ bayt} \approx 7.5\text{ GB}$$
   - Bir sunucuya aynı anda 10 adet bu tip görsel gönderildiğinde sunucu OOM (Out Of Memory) çöküşü yaşar.
   - **Çözüm:** `Image.MAX_IMAGE_PIXELS` ve dosya başlığı (header) pre-check doğrulaması.

2. **EXIF Oryantasyon Yanılsaması (Orientation Tag `0x0112`):**
   - Akıllı telefon kameraları dikey çekilen fotoğrafları piksel olarak dikey kaydetmez; yatay matris olarak kaydeder ve EXIF başlığına `Orientation = 6` (90° saat yönü) veya `8` (90° saat tersi) yazar.
   - Standart yükleyici bu etiketi yok saydığında, konvolüsyonel sinir ağı (CNN) insanı yan yatmış veya baş aşağı görür.
   - **Çözüm:** `ImageOps.exif_transpose()` ile fiziksel piksel matrisini transpoze etmek.

3. **Kesik ve Bozuk Ağ Akışları (`Truncated Images`):**
   - Zayıf mobil bağlantılarda dosya aktarımı yarıda kesilebilir. Standart Pillow `OSError: image file is truncated` hatasıyla çöker.
   - **Çözüm:** `ImageFile.LOAD_TRUNCATED_IMAGES = True` ile kurtarılabilir pikselleri tamamlamak.

4. **Kanal ve Renk Uzayı Kaosu (RGBA / CMYK / P / Grayscale $\to$ RGB):**
   - Model 3 kanal beklerken 4 kanallı RGBA geldiğinde `shape mismatch` oluşur. Şeffaf alfa kanalı silindiğinde siyah arka plan yerine şık bir beyaz mat kompoziti oluşturulmalıdır.

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          GELEN HAM GÖRSEL (Byte Stream / Dosya)          │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      GuvenliGorselYukleyici (Güvenlik ve Başlık Ön Kontrolü)                       │
    │  - Decompression Bomb Kontrolü: W * H <= maks_piksel_limiti (25 MP = 75 MB)                      │
    │  - Kesik Dosya Desteği (ImageFile.LOAD_TRUNCATED_IMAGES)                                          │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      EXIF DÜZELTME & RENK UZAYI NORMALİZASYON MOTORU                              │
    │  - EXIF Tag 0x0112 Okunur -> Fiziksel Piksel Transpozesi Yapılır (Dik Açı)                        │
    │  - RGBA/LA Şeffaflık -> Beyaz Arka Plan Alfa Mat Kompoziti -> Standart 3 Kanallı RGB             │
    │  - CMYK / L / Paletli (P) Modları -> sRGB Dönüşümü                                                │
    └───────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                 6-PANELLİ GÜVENLİ YÜKLEYİCİ TEŞHİS VE ONARIM PANELİ (Day 51)                      │
    └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#

---

### 📚 Kapsamlı Teknik Terimler ve Ek Kavramlar Sözlüğü

| Teknik Terim | İngilizce Karşılığı | Derinlemesine Açıklama ve Endüstriyel Önemi |
|---|---|---|
| **Görsel Dekompresyon Bombası** | *Decompression Bomb (Pixel Flood Attack)* | Küçük dosya boyutlu ancak açıldığında gigabaytlarca RAM tüketerek sistemi çökerten kötü niyetli görsel saldırısı. |
| **EXIF Oryantasyon Düzeltmesi** | *EXIF Orientation Transpose* | Akıllı telefon kameralarının görsele eklediği oryantasyon etiketini okuyup görüntüyü otomatik doğru yöne döndürme. |
| **Güvenli Bellek Akışı** | *In-Memory Buffer Streaming (`BytesIO`)* | Görselleri disk yerine doğrudan bellek akışında güvenli bayt sınırları içinde işleme. |
| **RGBA -> RGB Güvenli Dönüşüm** | *Alpha Channel Handling* | Saydam kanallı PNG görsellerini arkasına beyaz arka plan yerleştirerek kayıpsız 3 kanallı RGB formatına çevirme. |

---

## 2. Matematiksel Bellek ve Boyut Formülü

$$\text{Açılmış Bellek (Bayt)} = \text{Genişlik } (W) \times \text{Yükseklik } (H) \times \text{Kanal Sayısı } (C) \times 1\text{ bayt (uint8)}$$

Örnek $4000 \times 6000$ (24 Megapiksel) Görsel İçin:
$$\text{RAM} = 4000 \times 6000 \times 3 = 72,000,000 \text{ bayt} \approx 68.66\text{ MB}$$

---

## 🛠️ Dizin Yapısı

```
day-51-pillow-safe-image-loader/
├── LICENSE                          # Özel Lisans — Tüm Hakları Saklıdır
├── gereksinimler.txt                 # pillow, numpy, matplotlib, seaborn, pytest
├── ana_akis.py                      # 5 sınır senaryosunun uçtan uca güvenli yükleme betiği
├── README.md                        # 220+ satır teorik, matematiksel ve mimari dokümantasyon
├── src/
│   ├── __init__.py
│   ├── guvenli_yukleyici.py         # GuvenliGorselYukleyici (Decompression Bomb, EXIF & RGB Normalizasyonu)
│   ├── anomali_denetleyici.py       # GorselSaglikDenetleyicisi (En-Boy, Netlik, Varyans Denetimi)
│   └── gorsellestirici.py           # 6-Panelli Teşhis Panosu (Safe Image Loader Dashboard)
├── testler/
│   ├── __init__.py
│   └── test_safe_image_loader.py    # 7 adet birim test (Tümü Başarılı)
└── ciktilar/
    └── guvenli_yukleyici_paneli.png # 6 panelli yüksek çözünürlüklü teşhis panosu
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkların Kurulması
```bash
pip install -r gereksinimler.txt
```

### 2. Ana Akışın Çalıştırılması
```bash
python ana_akis.py
```

### 3. Birim Testlerin Koşturulması
```bash
pytest testler -v
```

---

## 📊 Sınır Senaryoları ve Güvenlik Yönetim Matrisi

| Senaryo | Tehdit / Risk | Güvenli Yükleyici Yanıtı | Çıktı Modu |
|---|---|---|---|
| **Normal JPEG** | Yok | Doğrudan RGB Tensöre Çevrilir | $(H, W, 3)$ RGB |
| **EXIF Rotasyonlu** | Yanlış Yönlendirme (Ters/Yan) | `ImageOps.exif_transpose()` ile Dik Çevrilir | $(W, H, 3)$ Doğru Yön |
| **Şeffaf RGBA PNG** | Model Boyut Hatası (4 Kanal) | Beyaz Arka Plan Alfa Mat Kompoziti | $(H, W, 3)$ RGB |
| **Kesik JPEG** | `OSError` Çökmesi | `LOAD_TRUNCATED_IMAGES` ile Kurtarma | $(H, W, 3)$ Tamamlanan RGB |
| **Decompression Bomb** | OOM / Sunucu Çöküşü | Başlıkta Boyut Kontrolü -> Engelleme | Reddedildi (`HATA`) |

---

## 🧪 Günün Alıştırması / Mini Görevi (Hands-on Challenge)

**Görev:** Yüklenen görsellerdeki gizlilik riskini ortadan kaldırmak için tüm GPS, Kamera Seri Numarası ve Cihaz EXIF etiketlerini sıfırlayan ve isteğe bağlı olarak kare kırpma (Center Crop) uygulayan bir **"Privacy Sanitization & Center Crop Pipeline"** geliştirmek.

**Tamamlanan Çözüm:**
```python
def gizlilik_ve_kare_kirp(img_pil: Image.Image, hedef_boyut: int = 224) -> Image.Image:
    """EXIF gizlilik verilerini sıfırlar ve en/boy oranını koruyarak kare merkez kırpma uygular."""
    # 1. EXIF verilerini temizle
    img_temiz = Image.new(img_pil.mode, img_pil.size)
    img_temiz.putdata(list(img_pil.getdata()))

    # 2. Merkezden kare kırpma
    w, h = img_temiz.size
    min_kenar = min(w, h)
    sol = (w - min_kenar) // 2
    ust = (h - min_kenar) // 2
    img_kirpilmis = img_temiz.crop((sol, ust, sol + min_kenar, ust + min_kenar))

    return img_kirpilmis.resize((hedef_boyut, hedef_boyut), Image.Resampling.BILINEAR)
```

---

## ❓ Gün Sonu Kontrol Noktası & Mentorluk Soru-Cevabı

> **Soru:** Bir mobil kullanıcı uygulamasından yüklenen bir profil fotoğrafı web tarayıcısında dik görünürken, PyTorch `Dataset` içine yüklendiğinde neden 90 derece saat yönünde dönmüş (yan yatmış) olarak tensöre dönüşür ve bu durum modelin doğruluğunu nasıl bozar?

> **Mentor Cevabı:**
> 1. **Tarayıcı vs Ham Piksel Ayrımı:** Modern web tarayıcıları EXIF başlığındaki `Orientation` etiketini (Tag `0x0112`) otomatik okur ve ekrana basarken CSS seviyesinde döndürür. Ancak Pillow'un standart `Image.open()` metodu ham pikselleri fiziksel dosya matrisinden okur; EXIF etiketini otomatik olarak piksele uygulamaz.
> 2. **Model Doğruluğuna Etkisi:** Evrişimsel Sinir Ağları (CNN) veya Vision Transformer'lar (ViT) insan yüzünün veya nesnenin dik durduğunu varsayan uzamsal filtrelere (spatial filters) sahiptir. Yan dönmüş bir yüz veya tekstil deseni öznitelik haritasını tamamen bozarak yanlış sınıflandırmaya sebep olur. Bu nedenle üretimde her zaman `ImageOps.exif_transpose(img)` uygulanmalıdır.

---

## 📜 Lisans

Bu proje **Özel Lisans — Tüm Hakları Saklıdır** kapsamındadır. Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas). İzin alınmaksızın ticari veya ticari olmayan projelerde kopyalanamaz, çoğaltılamaz veya dağıtılamaz.
