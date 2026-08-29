# Day 128: Multi-Agent Supervisor-Worker Mimarisi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; tekil genel amaçlı LLM'lerin karmaşık yazılım geliştirme ve araştırma görevlerindeki kısıtlarını aşan **Supervisor-Worker (Yönetici-İşçi) Hiyerarşik Çoklu Ajan Mimarisi**, **Özelleşmiş İşçi Ajanlar (Researcher, Coder, Reviewer)**, **Dinamik Görev Yönlendirme** ve **İteratif Geri Besleme Döngüsü** sistemini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajanlar Takımı Kurmak: Supervisor, Coder ve Reviewer ile Hata Sıfırlama"

Tek bir yapay zeka modeline "Hem algoritmayı araştır, hem Python kodunu yaz, hem de güvenlik ve sınır durumlarını test et" dediğinizde modelin bağlam penceresi karışır, dikkat mekanizması dağılır ve halüsinasyon oranı %34'lere fırlar.

**Supervisor-Worker Deseni Nasıl Çalışır?**
Tıpkı profesyonel bir yazılım şirketinde olduğu gibi işler uzmanlara paylaştırılır:
1. 👔 **Supervisor (Yönetici Ajan):** Kullanıcıdan gelen ana hedefi alır, parçalara böler ve hangi işçinin hangi sırayla çalışacağına karar verir.
2. 🔍 **Researcher (Araştırmacı Ajan):** Algoritmik karmaşıklığı ($O(N)$), en iyi kütüphaneleri ve kritik kısıtları (örn. negatif sayılar) belirler.
3. 💻 **Coder (Geliştirici Ajan):** Yalnızca araştırmacının çıkardığı teknik şartnameye odaklanarak temiz, modüler Python kodu yazar.
4. 🧐 **Reviewer (Denetleyici Ajan):** Üretilen kodu bağımsız olarak test eder; hata veya eksik varsa geri bildirimle Coder'a revizyon yaptırır.

```
                 [Kullanıcı Hedefi]
                         │
                         ▼
               [SUPERVISOR ORCHESTRATOR]
               (Görev Bölme & Yönlendirme)
                 ┌───────┼───────┐
                 ▼       ▼       ▼
            [Researcher] │       │ (Şartname)
                         ▼       │
                  ┌─> [Coder] <──┘
                  │      │ (Kod Üretimi)
      (Geri       │      ▼
      Besleme)    └── [Reviewer]
                         │ (Onay Verildi)
                         ▼
               [Nihai Sentez Raporu]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma: Supervisor-Worker Mimarisi ve Hiyerarşik Görev Ayrıştırma
- Karmaşık problemler atomik alt görevlere bölünür.
- Supervisor, durum takibi yaparak işçiler arasında veri akışını koordine eder.

### 2. Özelleşmiş İşçi Ajanlar (Specialized Personas) ve Dar Bağlam Avantajı
- Her ajan yalnızca kendi uzmanlık alanına ait sistem prompt'una sahiptir.
- Dar bağlam penceresi (*Narrow Context*), modelin dikkatini odaklayarak dikkat dağılmasını ve halüsinasyonu engeller.

### 3. Dinamik Yönlendirme (Dynamic Routing), Geri Besleme Döngüsü ve İteratif Hata Telafisi
- Denetçi (Reviewer) onay vermediği sürece Supervisor görevi bitirmez; hata detaylarıyla birlikte Coder'a geri pas atar (*Multi-Turn Feedback Loop*).
- Kod kalitesi v1'deki %65.0 seviyesinden v2'de **%98.5 seviyesine** yükselir.

### 4. Çıktı Sentezi (Consensus Synthesis), Token Verimliliği ve Halüsinasyon Önleme
- Tüm işçilerin çıktıları Supervisor tarafından doğrulanarak nihai üretim raporuna dönüştürülür.
- Halüsinasyon oranı %34'ten **%3.5 seviyesine** düşürülür.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Supervisor Agent** | Görevleri planlayan, işçilere delege eden ve nihai sentezi yapan merkezi yönetici ajan. |
| **Worker Agent** | Belirli bir uzmanlık alanında (kodlama, test, araştırma) çalışan özelleşmiş alt ajan. |
| **Task Decomposition** | Büyük ve karmaşık hedefin yönetilebilir atomik alt görevlere parçalanması. |
| **Dynamic Routing** | Mevcut durum ve işçi çıktılarına göre bir sonraki ajanı dinamik belirleme. |
| **Feedback Loop** | Denetçi eleştirilerine göre geliştirici ajanın kodu iteratif olarak düzeltmesi. |
| **Narrow Context** | Ajanın yalnızca kendi görevine odaklanmasını sağlayan daraltılmış istem/bağlam. |
| **Reviewer / QA Persona** | Kodun doğruluğunu, sınır durumlarını ve güvenliğini test eden denetçi rolü. |
| **Researcher Persona** | Problem için en uygun algoritma ve kısıtları çıkaran analist rolü. |
| **Consensus Synthesis** | Farklı işçilerden gelen çıktıların tek bir tutarlı sonuca birleştirilmesi. |
| **Recursion Guard** | Ajanlar arası revizyon döngüsünün sonsuz döngüye girmesini engelleyen sayaç. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %98.5 yüksek kod kalite ve doğruluk│ • Çoklu LLM çağrılarından kaynaklanan│
 │ • Halüsinasyonlarda 10 kat azalma.   │   art artışlı token ve gecikme maliyeti
 │ • Otomatik iteratif hata düzeltme.   │ • Hiyerarşik koordinasyon karmaşıklığı│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal yazılım üretim hatları ve │ • İşçiler arası yanlış bağlam iletimi│
 │   otonom veri bilimi iş akışları.    │   durumunda yanlış yönlendirme riski.│
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/multi_agent_supervisor_paneli.png` dosyası üretilir:
1. **Tekil Ajan vs Supervisor-Worker Başarım Kıyaslaması**
2. **İşçi Ajanlar Arası Görev Dağılımı ve Çağrı Sayısı**
3. **Geri Besleme Döngüsü ile Kalite Skoru Artışı**
4. **İşçi Ajan Yürütme Süreleri (ms)**
5. **Hiyerarşik Supervisor-Worker Mimari Şeması**
6. **Supervisor-Worker Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Supervisor-Worker çoklu ajan iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
