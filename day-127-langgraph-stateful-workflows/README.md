# Day 127: LangGraph Durumsal Çizge (StateGraph) & Human-in-the-Loop İş Akışları

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; doğrusal LLM zincirlerinin (Linear Chains) sınırlarını aşan **LangGraph StateGraph Mimarisi**, **Durum İndirgeyiciler (State Reducers)**, **Koşullu Kenarlar (Conditional Edges)**, **Bellek Kontrol Noktaları (Checkpointing / Time Travel)** ve **İnsan-Döngüde Kesintisi (Human-in-the-Loop Interrupts)** sistemini sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Ajan Akışlarını Çizgeye Dökmek: LangGraph ve İnsan Onayı"

Eski nesil LangChain veya doğrusal LLM hatlarında akış yalnızca tek yönde ($A \to B \to C$) ilerleyebilirdi. Eğer $B$ aşamasında kod veya çıktı hatalıysa başa dönüp kendini düzeltme (**Self-Correction Loop**) şansı yoktu. Ayrıca kritik finansal işlemlerde (örn. 10.000 TL para iadesi) araya bir insanın girip onay vermesi imkansızdı.

**LangGraph StateGraph Mimarisi Neyi Değiştirir?**
1. 🕸️ **Düğümler (Nodes):** Her biri bağımsız birer fonksiyon olan ajan ve araç düğümleri (`TalepAyristirici`, `RiskDegerlendirici`, `OdemeIadesi`).
2. 🔄 **Döngüler ve Koşullu Kenarlar (Conditional Edges):** İşlem başarılı olana kadar geri besleme döngüsüyle çalışabilir veya riske göre farklı yollara dallanabilir.
3. 📦 **Durum Sözlüğü (State):** Tüm düğümler tek bir paylaşılan durum nesnesini okur ve günceller (`State Reducer`).
4. ⏸️ **Human-in-the-Loop (HITL):** Kritik bir eşik aşıldığında (örn. Yüksek Risk) çizge duraklar (**Interrupt**), insan onayı gelene kadar durumu dondurur ve onaylanınca kaldığı yerden devam eder.
5. ⏪ **Zaman Yolculuğu (Time Travel / Checkpointing):** Her adımın anlık görüntüsü saklanır; istenirse geçmiş bir adıma geri sarılabilir (**Rollback**).

```
               [START: TalepAyristirici]
                          │
                          ▼
                [RiskDegerlendirici]
                 ┌────────┴────────┐
                 │                 │
        (Risk <= 0.70)       (Risk > 0.70)
                 │                 │
                 │                 ▼
                 │       [INTERRUPT: InsanOnayi]
                 │           ┌─────┴─────┐
                 │        (Onay)     (Red)
                 ▼           ▼           ▼
           [OdemeIadesi] ───┘      [TalepReddi]
                 │                       │
                 └───────────┬───────────┘
                             ▼
                  [BilgilendirmeEpostasi]
                             │
                             ▼
                           [END]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & LangGraph StateGraph Mimarisi
- **Doğrusal Zincirler (DAG):** Hata anında tüm akış kırılır; döngü ve geri besleme mekanizması yoktur.
- **StateGraph Çizgeleri:** Döngüsel grafikleri (*Cyclic Graphs*) birinci sınıf vatandaş olarak destekler. Ajan çıktıyı beğenmezse önceki düğüme geri dönebilir.

### 2. Durum İndirgeyiciler (State Reducers), Mesaj Kanalları ve Tip Güvenliği
- Düğümler tüm durumu ezmez; yalnızca ürettikleri güncellemeleri döndürür.
- `DurumIndirgeyici`, mesaj listelerini birbirine eklerken (`append`) skaler değişkenleri güvenle günceller.

### 3. Koşullu Kenarlar (Conditional Edges), Dinamik Yönlendirme ve Döngü Güvenliği
- `add_conditional_edges` ile yönlendirici fonksiyonlar (`router_fn`) çalışma anındaki risk ve başarı durumuna göre rotayı dinamik olarak belirler.
- Sonsuz döngüleri engellemek için `max_tekrar` (Recursion Limit) koruması devrededir.

### 4. Kontrol Noktaları (Checkpointing / Time Travel) ve Human-in-the-Loop (HITL) Kesinti Yönetimi
- Kritik kararlarda çizge çalışmayı keser (`tamamlandi=False, kesinti_noktasi="InsanOnayi"`).
- Yetkili kullanıcı durumu inceleyip onay verdikten sonra çizge kaldığı adımdan sorunsuz devam eder.

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **LangGraph** | LLM'lerle döngülü ve durumsal çoklu ajan akışları kurmaya yarayan çizge çerçevesi. |
| **StateGraph** | Düğümler ve kenarlar arasında paylaşılan bir durum (State) taşıyan çizge motoru. |
| **State Reducer** | Düğüm çıktılarındaki değişiklikleri mevcut duruma güvenle birleştiren indirgeyici. |
| **Conditional Edge** | Durum içeriğine göre hangi sıradaki düğüme gidileceğini belirleyen koşullu kenar. |
| **Human-in-the-Loop (HITL)** | Ajanın kritik işlemlerde durup insan onayını beklemesini sağlayan kesinti yapısı. |
| **Checkpoint** | Bir çizge adımının tüm değişkenleriyle birlikte alınan anlık durum görüntüsü (Snapshot). |
| **Time Travel / Rollback** | Kaydedilmiş geçmiş bir kontrol noktasına dönerek akışı yeniden başlatabilme. |
| **Recursion Limit** | Döngülü çizgelerin sonsuz döngüye girmesini engelleyen maksimum tekrar sayısı kısıtı. |
| **Entry Point** | Çizgenin `START` sinyaliyle ilk olarak tetikleneceği başlangıç düğümü. |
| **END Node** | Çizgenin tüm iş akışını başarıyla tamamlayıp sonlandığı özel terminal düğüm. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Döngülü geri besleme ve telafi.    │ • Durum nesnesi büyüdükçe artan      │
 │ • %100 güvenli insan onayı (HITL).   │   bellek ve serileştirme yükü.       │
 │ • Anında zaman yolculuğu ve geri sar-│ • Karmaşık çoklu düğümlerde hata     │
 │   ma (Time Travel / Rollback).       │   ayıklama ve çizge takibinin zorluğu│
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Kurumsal bankacılık, müşteri deste-│ • Yanlış yapılandırılmış koşullu ke- │
 │   ği ve çoklu ajan orkestrasyonu.    │   narlarda döngü kilitlenmesi riski. │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/langgraph_paneli.png` dosyası üretilir:
1. **Doğrusal Zincir vs LangGraph Başarım Oranları**
2. **İş Akışı Düğüm Geçişleri ve Risk Seviyesi (%)**
3. **Checkpoint Durum Büyümesi ve Kanal İndirgeme**
4. **Karar Mekanizması Dağılımı (Düşük vs Yüksek Risk)**
5. **LangGraph StateGraph ve Koşullu Yönlendirme Şeması**
6. **LangGraph StateGraph Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# LangGraph durumsal iş akışını çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
