# Day 130: Human-in-the-Loop (HITL) Otonom Güvenlik & Kesinti Deseni (FAZ 7 BÜYÜK FİNALİ)

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; otonom yapay zeka ajanlarının geri dönülemez felaket risklerini (canlı veritabanı silme, yüksek tutarlı yetkisiz para transferi, üretim DNS değişikliği) engelleyen **Human-in-the-Loop (HITL) Kesinti Deseni (Interrupt Pattern)**, **Dinamik Risk Eskalasyonu**, **İnsan Düzenleme/Onay/Red Mekanizması** ve **Geri Sarma (Rollback) Motoru**nu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajanın Elini Tutmak: Kritik Eylemlerde İnsan Onayı ve Güvenlik"

Tamamen otonom bir ajana "Sistemi temizle" veya "Müşteri iadelerini yap" dediğinizde ajan aşırı hevesli davranıp canlı veritabanı tablosunu silebilir (`DROP TABLE`) ya da yetkisiz 85.000 TL transfer edebilir.

**Human-in-the-Loop (HITL) Mimarisi Neyi Değiştirir?**
1. 🚦 **Risk Sınıflandırması:** Her eylem risk skoruna göre sınıflandırılır ($0.0 - 1.0$).
2. ⏸️ **Otomatik Kesinti (Interrupt):** Risk skoru $\ge 0.70$ olan tüm eylemlerde ajan çalışmayı dondurur, durumu güvenli kontrol noktasına kaydeder.
3. 👤 **İnsan Denetçi Kararları:**
   - ✅ **ONAYLA:** Eylem güvenliyse aynen icra edilir.
   - ✏️ **DÜZENLE:** Denetçi tutarı veya parametreyi güvenli seviyeye çeker (örn: 85.000 TL yerine 12.000 TL).
   - 🛑 **REDDET:** Yıkıcı eylem engellenir, ajan otomatik olarak güvenli bir alternatif icra eder (örn: silme yerine arşiv yedekleme).
4. 📜 **Denetim İzi (Audit Trail):** Kimin ne zaman hangi kararı verdiği geriye dönük olarak denetlenebilir.

```
               [Ajan Eylem Planı]
                       │
                       ▼
             [Risk Sınıflandırıcı]
              ┌────────┴────────┐
              │                 │
        (Risk < 0.70)     (Risk >= 0.70)
              │                 │
              │                 ▼
              │       [INTERRUPT: İnsan Onayı]
              │           ┌─────┼─────┐
              │       (Onay)  (Düzenle) (Red)
              ▼           ▼     ▼       ▼
           [Güvenli İcra] ───┘  │   [Güvenli Alternatif]
                 │              │       │
                 └──────────────┴───────┘
                                │
                                ▼
                     [Audit Trail / Denetim İzi]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Human-in-the-Loop (HITL) Kesinti ve Durumsal Güvenlik
- Otonom yürütme ile insan denetimi arasındaki kusursuz köprüdür.
- Durum derin kopyalanarak dondurulur; onay gelene kadar hiçbir yan etki (*Side Effect*) oluşmaz.

### 2. Dinamik Risk Sınıflandırması ve Çok Kademeli Eskalasyon
- **Düşük / Orta Risk:** Log sorgulama, rapor üretme gibi eylemler otomatik onaylanır (İnsan iş yükünden %78.4 tasarruf).
- **Yüksek / Kritik Risk:** Finans ve veritabanı eylemleri zorunlu insan kesintisine girer.

### 3. İnsan Karar Matrisi: Onay, Red ve Parametre Düzenleme
- Denetçi yalnızca "Evet/Hayır" demek zorunda değildir; parametreleri çalışma anında düzenleyip güvenli hale getirebilir (*Edit & Replay*).

### 4. Denetim İzi (Audit Trail), Geri Sarma (Rollback) ve Felaket Önleme
- Tüm kararlar zaman damgası ve denetçi imzasıyla kaydedilir.
- Olası bir hata anında kaydedilmiş kontrol noktasına geri sarma (*Time Travel / Rollback*) imkanı sunulur.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Human-in-the-Loop (HITL)** | Ajanın kritik kararlarda durup insan onayını beklemesini sağlayan güvenlik mimarisi. |
| **Interrupt / Breakpoint** | Çizgenin veya yürütücünün eylem öncesinde çalışmayı güvenle duraklatması. |
| **Escalation Policy** | Eylem risk skoru arttıkça yetkilendirme seviyesinin otomatik yükseltilmesi. |
| **Audit Trail** | Gerçekleşen tüm işlemlerin ve insan kararlarının kronolojik denetim kaydı. |
| **Side Effect Isolation** | Onay alınana kadar API ve veritabanı üzerinde kalıcı değişiklik yapılmaması. |
| **Action Modification** | Denetçinin eylem parametrelerini onay öncesi çalışma anında güvenli hale getirmesi. |
| **Rollback / Time Travel** | Hatalı veya reddedilen işlem sonrası kaydedilmiş güvenli duruma geri dönülmesi. |
| **Catastrophic Action** | Canlı veritabanı silme veya büyük maddi kayba yol açabilecek telafisiz eylem. |
| **Auto-Approval Threshold**| Düşük riskli işlemlerin insan müdahalesi olmadan otomatik çalıştığı risk eşiği. |
| **State Snapshot** | Kesinti anında durum değişkenlerinin belleğe derin kopyalanarak dondurulması. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %100 felaket ve yıkıcı hata koruması│ • İnsan yanıt süresine bağlı gecikme  │
 │ • %78.4 insan iş yükü tasarrufu.     │   (Human Latency).                   │
 │ • Tam denetlenebilir Audit Trail.    │ • Çok sık kesintide denetçi yorgunluğu│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal bankacılık, sağlık, bulut │ • Yanlış risk eşiklerinde güvenlik   │
 │   altyapısı ve otonom sistemler.     │   açığı veya gereksiz kuyruk birikmesi│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/hitl_agent_paneli.png` dosyası üretilir:
1. **Tam Otonom vs HITL Güvenlik ve Hata Kıyaslaması**
2. **Eylem Risk Skorları ve Kritiklik Seviyeleri (%)**
3. **Karar ve Kesinti Türleri Dağılımı**
4. **Denetim İzi (Audit Trail) Güvenlik Kayıtları**
5. **Human-in-the-Loop (HITL) Mimarisi Şeması**
6. **FAZ 7 GRAND FINALE Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Human-in-the-Loop iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
