# 🚗 Tesla FSD Otonom Sürüş | Gün 73: OTA (Over-the-Air) Güncelleme Mimarisi: A/B Bölümlendirme ve Geri Alma (Rollback)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![OTA](https://img.shields.io/badge/OTA-Over--the--Air%20Update-red.svg?style=flat-square)](https://www.tesla.com/)
[![Partitioning](https://img.shields.io/badge/Architecture-Seamless%20A%2FB%20Dual--Slot-blue.svg?style=flat-square)](https://systemd.io/BOOT_LOADER_SPECIFICATION/)
[![Rollback](https://img.shields.io/badge/Safety-Zero--Brick%20Auto--Rollback-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"73. günümüze hoş geldin stajyer!  
> Tesla'nın otomotiv sektörünü kökünden değiştiren en devrimci özelliği, aracın servise gitmeden gece yarısı Wi-Fi üzerinden yeni otonom sürüş özellikleri kazanmasını sağlayan **OTA (Over-the-Air) Güncelleme Sistemidir**.  
> Ancak milyonlarca araca uzaktan yazılım gönderirken tek bir arabanın bile 'brick' olması (kullanılamaz hale gelmesi) kabul edilemez!  
> Bu yüzden Tesla **A/B Çift Bölümlendirme (Dual-Slot Partitioning)** mimarisini kullanır:  
> 1. **Kesintisiz Arka Plan Yazma:** Araç sürüş halindeyken Slot A çalışır; yeni güncelleme sessizce pasif olan Slot B'ye yazılır.  
> 2. **Sıfır Kesinti:** Sürücü park edip aracı kilitlediğinde aktif slot Slot B olarak işaretlenir (`bootctl mark-good`).  
> 3. **Otomatik Geri Alma (3-Fault Rollback):** Eğer yeni yazılım açılışta donarsa veya 3 ardışık boot hatası verirse, donanım anında eski çalışan Slot A'ya geri döner ($0\text{ ms}$ gecikme).  
> 4. **Sıfır Brick Riski:** Aracın servise çekilme ihtimali matematiksel ve donanımsal olarak sıfırlanır!  
> Bugün Tesla'nın kablosuz yazılım güncelleme ve otomatik kurtarma motorunu inşa ediyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. A/B Slot Durum Geçiş Modeli

$$\text{ActiveSlot}(t+1) = \begin{cases} \text{PassiveSlot}(t), & \text{OTA Güncellemesi Tamamlandı \& Reboot} \\ \text{PreviousSlot}, & N_{\text{failed\_boots}} \ge 3 \ (\text{Otomatik Rollback}) \\ \text{ActiveSlot}(t), & \text{Boot Başarılı (mark-good)} \end{cases}$$

### 2. Geri Alma (Rollback) Tetikleme Kriteri

$$\text{ROLLBACK} = \text{TRUE} \iff N_{\text{failed\_boots}} \ge 3$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Dünya genelindeki 6+ milyon Tesla aracına fiziksel servise ihtiyaç duymadan kablosuz yeni FSD sürümleri yüklemek ve güncelleme hatalarında sıfır araç kilitlenmesi garantisi sunmak için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Araç Kilitlenmesi (Bricking):** Elektrik kesintisi veya bozuk yazılım durumunda eski çalışan slota anında geri dönülerek aracın yolda kalmasını engelledi.
- **Sürüş Esnasında Güncelleme:** Güncelleme dosya yazımının arka planda sürüşü engellemeden yapılması sağlandı.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Depolama Alanı:** Çift işletim sistemi (Slot A + Slot B) flash bellekte iki kat disk alanı (örneğin $2 \times 16\text{ GB}$) kaplar.

### 4. Alternatifler Nelerdir? (Alternatives)
- **Tek Slot Üzerine Yazma (Klasik Yöntem):** Güncelleme sırasında güç kesilirse cihaz brick olur (Otomotivde KABUL EDİLEMEZ).
- **Kurtarma Modu (Recovery Mode):** Kullanıcının manuel müdahale etmesini gerektirir.

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **OTA (Over-the-Air)** | Kablosuz hücresel (LTE/5G) veya Wi-Fi ağı üzerinden uzaktan yazılım güncelleme teknolojisi. |
| **A/B Partitioning** | İki özdeş işletim sistemi bölümünün (Slot A ve B) sırayla güncellenip kullanıldığı mimari. |
| **Rollback** | Yeni yazılım başarısız olduğunda sistemin otomatik olarak önceki çalışan sürüme dönmesi. |
| **bootctl mark-good** | Yeni slotun başarıyla başladığını doğrulayan ve geri almayı devreden çıkaran komut. |
| **Bricking** | Hatalı yazılım yüklemesi nedeniyle bir elektronik cihazın tamamen kullanılamaz hale gelmesi. |
| **Passive Slot** | Arka planda güncelleme paketinin yazıldığı o an aktif olmayan işletim sistemi bölümü. |
| **Delta Update** | Tüm işletim sistemi yerine sadece değişen ikili farkların (Binary Diff) indirilmesi. |
| **Watchdog Panic** | Açılış sırasında sistemin yanıt vermemesi durumunda donanımsal reset tetikleyen sayaç. |
| **Rootfs** | Linux işletim sisteminin tüm temel dosyalarını barındıran kök dosya sistemi bölümü. |
| **Flash Wear Leveling** | A/B bölümlerine yazma yaparken flash bellek hücrelerinin eşit yıpranmasını sağlayan dengeleme. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 sıfır brick riski ve otomatik 3-hatada rollback| • Çift işletim sistemi nedeniyle 2x flash bellek alanı|
| • Arka planda kesintisiz indirme ve kurulum           | • Delta güncelleme sıkıştırma algoritmalarının        |
| • 2 µs ultra hızlı durum makinesi geçişi              |   ek CPU yükü                                         |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Tüm filoya 24 saat içinde sıfır maliyetle yeni yapay| • Düşük hücresel çekim alanlarında OTA indirmelerinin |
|   zeka FSD modelleri ve eğlence özellikleri dağıtımı  |   kesintiye uğraması (Resume özelliği gerektirir)     |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla OTA A/B Slot Durum Makinesi Şeması

```
[ Mevcut Durum: Slot A Aktif (v2026.4.1) ]
                   |
                   | 1. Arka Planda İndir & Slot B'ye Yaz
                   v
[ Slot B Hazırlandı (v2026.8.0) -> PENDING REBOOT ]
                   |
                   | 2. Araç Kilitlendi -> Slot B'den Başlat
                   v
[ Slot B Boot Ediliyor... ]
         |                         |
         | Başarılı                | 3 Ardışık Çökme / Hata
         v                         v
[ bootctl mark-good ]       [ OTOMATİK ROLLBACK ]
- Slot B Kalıcı Onaylandı   - Slot A'ya Geri Dönüldü (v2026.4.1)
- Sıfır Hata                - Sıfır Brick Riski (%100 Güvenli)
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana OTA A/B Rollback simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
