# 🚗 Tesla FSD Otonom Sürüş | Gün 72: Güvenli Önyükleme (Secure Boot), TPM 2.0 ve Kriptografik Ürün Yazılımı Doğrulama

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![Secure Boot](https://img.shields.io/badge/Security-Hardware%20Root%20of%20Trust-red.svg?style=flat-square)](https://www.tesla.com/)
[![Crypto](https://img.shields.io/badge/Crypto-RSA--4096%20%2F%20ECDSA%20P--384-blue.svg?style=flat-square)](https://trustedcomputinggroup.org/)
[![Tamper-Proof](https://img.shields.io/badge/Integrity-dm--verity%20Rootfs%20Tree-green.svg?style=flat-square)](https://www.iso.org/)

---

## 👨‍🏫 Mentor'dan Stajyere Hoş Geldin Notu

> *"72. günümüze hoş geldin stajyer!  
> Otonom bir aracın kontrolü hacker'ların eline geçerse felaket olur!  
> Bir saldırganın aracın flash belleğini söküp içine zararlı yazılım (Rootkit) yüklemesini engellemek için Tesla, donanım seviyesinde **Güvenli Önyükleme (Secure Boot) ve TPM 2.0 Güven Zinciri (Chain of Trust)** uygular:  
> 1. **Hardware Root of Trust (RoT):** İşlemci çipine üretim bandında lazerle yazılan (eFuse OTP) genel anahtar asla değiştirilemez.  
> 2. **Kademeli Güven Zinciri:**  
>    - Stage 1: Çip içi ROM $\to$ Stage 2: U-Boot RSA-4096 İmzası Doğrulanır.  
>    - Stage 2: U-Boot $\to$ Stage 3: Linux Çekirdeği ECDSA P-384 İmzası Doğrulanır.  
>    - Stage 3: Linux Çekirdeği $\to$ Stage 4: dm-verity ile tüm dosya sistemi doğrulanır.  
> 3. **Sabit Zamanlı (Constant-Time) Hash Karşılaştırması:** Yan kanal zamanlama saldırılarına (Timing Attacks) karşı XOR temelli sabit süreli karşılaştırma yapılır.  
> 4. **Yetkisiz Yazılım Kalkanı:** İmzası tutmayan veya tahrif edilmiş tek bir bit bile tespit edilirse sistem kilitlenir ve araç güvenli moda geçer.  
> Bugün Tesla'nın siber güvenlik kalesinin temel taşı olan Secure Boot motorunu kodluyoruz!"*

---

## 📐 Matematiksel ve Donanımsal Modelleme

### 1. Kademeli Güven Zinciri Doğrulama Kanunu

$$\text{ChainOfTrust} = \bigwedge_{i=1}^4 \text{VerifySignature}\left( \text{Stage}_i, \ \mathbf{K}_{\text{public}} \right)$$

### 2. Sabit Zamanlı (Constant-Time) Güvenli Eşleşme Fonksiyonu

$$\text{Diff}(x, y) = \bigvee_{j=1}^{64} \left( x_j \oplus y_j \right) \implies \text{Valid} \iff \text{Diff}(x, y) == 0$$

### 3. SHA-256 ve ECDSA P-384 Kriptografik İmza Doğrulama

$$H = \text{SHA256}(\text{FirmwareImage}), \quad \text{Verify}_{\text{ECDSA}}(H, \ \sigma, \ \mathbf{Q}_{\text{Tesla}}) \in \{\text{True}, \text{False}\}$$

---

## 🏛️ 4 Zorunlu Mimari Analiz Bölümü

### 1. Neden Kullanıldı? (Why Used)
Araç bilgisayarına yetkisiz yazılım yüklenmesini, modifikasyonları (jailbreak) ve tedarik zinciri saldırılarını donanım seviyesinde engellemek için kullanıldı.

### 2. Neyi Çözdü? (What It Solved)
- **Kötü Amaçlı Yazılım Enjeksiyonu:** İmzalanmamış veya modifiye edilmiş hiçbir çekirdek/sürücünün çalışmasına izin vermedi.
- **Sürüm Düşürme (Anti-Rollback):** Bilinen güvenlik açığı olan eski yazılım sürümlerine geri dönülmesini eFuse sayaçları ile engelledi.

### 3. Sınırları ve Boşlukları (Limitations & Gaps)
- **Geliştirici İzolasyonu:** Güvenlik katmanı nedeniyle hata ayıklama (JTAG) portları kapatılır (Geliştirici modunda özel kriptografik sertifikalar gerekir).

### 4. Alternatifler Nelerdir? (Alternatives)
- **Yazılımsal Doğrulama (Software Hash Check):** İşletim sistemi açıldıktan sonra hash kontrolü yapmak güvenli değildir (Çekirdek zaten ele geçirilmiş olabilir).

---

## 📖 10 Terimli Mühendislik Sözlüğü

| Terim | Tanım |
|---|---|
| **Secure Boot** | Açılış anındaki tüm yazılım parçalarının kriptografik olarak imzalanıp doğrulanması süreci. |
| **Root of Trust (RoT)** | Güven zincirinin en temelinde yer alan, donanımsal olarak değiştirilemez güven kökü (eFuse ROM). |
| **TPM 2.0** | Kriptografik anahtarları, sertifikaları ve platform ölçümlerini saklayan özel güvenlik çipi. |
| **Chain of Trust** | Her açılış aşamasının bir sonraki aşamanın imzasını doğrulayarak kontrolü devrettiği zincir. |
| **dm-verity** | Linux çekirdeğinin kök dosya sistemindeki her bloğu Merkle ağacı ile anlık doğrulayan güvenlik modülü. |
| **eFuse OTP** | Çip üretimi sırasında kalıcı olarak yakılan ve bir daha asla değiştirilemeyen tek yazımlık bellek. |
| **Constant-Time Compare** | Karşılaştırma süresinin veri içeriğine göre değişmesini engelleyen kriptografik algoritma. |
| **Anti-Rollback** | Eski güvenlik açıklı sürümlere geri dönülmesini engelleyen donanımsal sürüm sayacı. |
| **ECDSA P-384** | 384-bit eliptik eğri dijital imza algoritması. |
| **Timing Attack** | Bir parolanın veya hash'in eşleşme süresindeki mikrosaniyelik farkları ölçerek anahtarı çözme saldırısı. |

---

## 📊 ASCII SWOT Matrisi

```
+-------------------------------------------------------+-------------------------------------------------------+
|                    GÜÇLÜ YÖNLER (STRENGTHS)            |                   ZAYIF YÖNLER (WEAKNESSES)           |
+-------------------------------------------------------+-------------------------------------------------------+
| • %100 silikon seviyesinde Root of Trust koruması     | • Anahtar sızıntısı durumunda donanımsal eFuse        |
| • Sabit zamanlı XOR ile sıfır yan kanal saldırısı     |   değişimi gerekliliği                                |
| • 3 µs ultra hızlı kriptografik doğrulama             |                                                       |
+-------------------------------------------------------+-------------------------------------------------------+
|                  FIRSATLAR (OPPORTUNITIES)            |                    TEHDİTLER (THREATS)                |
+-------------------------------------------------------+-------------------------------------------------------+
| • Otonom araç filolarında regülasyon uyumu (UN R155 / | • Kuantum bilgisayarların klasik RSA-4096 imza        |
|   R156 Siber Güvenlik Standardı)                      |   sistemlerini gelecekte tehdit etmesi                |
+-------------------------------------------------------+-------------------------------------------------------+
```

---

## 🏗️ Tesla Secure Boot Güven Zinciri Akış Şeması

```
[ Donanımsal OTP eFuse ROM (Hardware Root of Trust) ]
                         |
                         | 1. Doğrulama: SHA-256 + RSA-4096
                         v
          [ Stage 2: U-Boot Bootloader ]
                         |
                         | 2. Doğrulama: ECDSA P-384
                         v
       [ Stage 3: Özel Linux Çekirdeği (Kernel) ]
                         |
                         | 3. Doğrulama: dm-verity Merkle Ağacı
                         v
      [ Stage 4: Kök Dosya Sistemi (Rootfs Verified) ]
                         |
                         v
            [ GÜVENLİ FSD SÜRÜŞ BAŞLADI ]
```

---

## 💻 Kullanım ve Test

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# Birim testleri çalıştırın
pytest testler/

# Ana Secure Boot simülasyonunu ve tanı panelini üretin
python ana_akis.py
```

---

## 📄 Lisans

Telif Hakkı (c) 2026 Seydi Eryılmaz ([@seydivakkas](https://github.com/seydivakkas)) — Tüm Hakları Saklıdır.  
Yalnızca görüntüleme ve eğitim amaçlıdır. Ticari kullanılamaz, kopyalanamaz.
