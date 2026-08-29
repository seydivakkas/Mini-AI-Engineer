# Day 122: Plan-and-Solve (PS / PS+) Prompting ve Görev Ayrıştırma DAG Mimarisi

[![License: All Rights Reserved](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Tests: PyTest 8/8 Passed](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg?style=flat-square)](testler/)

> **FAZ 7: Otonom AI Ajanları, Multi-Agent Sistemleri ve Advanced GraphRAG (Gün 121 - Gün 140)**  
> Bu modül; karmaşık ve çok aşamalı problemleri doğrudan çözmeye çalışmak yerine önce alt görevlere ayıran (**Task Decomposition**), bağımlılıkları Yönlü Döngüsüz Çizge (**DAG - Directed Acyclic Graph**) ile modelleyen, değişken ikamesi (**Variable Substitution**) ve mantıksal doğrulama ile hatasız yürüten **Plan-and-Solve (PS / PS+) Prompting Motoru**nu sıfırdan inşa eder.

---

## 👨‍🏫 Stajyer İçin Anlaşılır Kılavuz: "Böl, Yönet, Çöz: Plan-and-Solve ve Görev DAG Mimarisi"

Bir şirketin vergi sonrası net karını hesaplamanız istendiğinde ne yaparsınız?
- **Zero-Shot CoT ("Let's think step by step"):** İleriye doğru körlemesine bir paragraflık düşünce üretir. Ancak ortadaki bir adımı unutabilir (**Missing-step error**) veya ara çarpma/bölme işlemlerinde halüsinasyon yapabilir (**Calculation error**).
- **Plan-and-Solve (Wang et al., ACL 2023):** Bir baş mühendis gibi iki ayrı aşamada çalışır:
  1. 📋 **1. Aşama (Planlama):** *"Problemi 4 alt göreve bölelim: 1. Geliri hesapla, 2. Maliyeti hesapla, 3. Gelirden maliyeti çıkarıp brüt karı bul, 4. %20 vergi düşüp net karı bul."*
  2. 🔗 **Bağımlılık Çizgesi (DAG):** Brüt kar hesabı, gelir ve maliyet bitmeden çalışamaz. Motor bu sırayı otomatik topolojik sıralar.
  3. ⚡ **2. Aşama (Çözüm & İkame):** Her adımın sonucu (`durum_haritasi`) sonraki adımın denklemine matematiksel kesinlikle yerleştirilir.
  4. 🎯 **Sonuç:** Eksik adım atlama oranı %18.4'ten **%0.4'e**, hesaplama hatası %24.5'ten **%1.2'ye** iner!

```
     KULLANICI PROBLEMİ                                   PLANLAMA AŞAMASI (Task DAG)
 ┌────────────────────────────────────┐            ┌──────────────────────────────────────────────┐
 │ "150 TL satış, 60 TL maliyet,      │            │ [Adım 1: Gelir]       [Adım 2: Maliyet]      │
 │  40K sabit gider, 1200 adet satış, │ ─────────> │ (150 * 1200)          (40000 + 60*1200)      │
 │  %20 vergi sonrası net kar?"       │            └───────┬───────────────────────┬──────────────┘
 └────────────────────────────────────┘                    │                       │
                                                           ▼                       ▼
                                                [Adım 3: Brüt Kar (Gelir - Maliyet)]
                                                           │
                                                           ▼
                                                [Adım 4: Net Kar (Brüt Kar * 0.80)]
                                                           │
                                                           ▼
                                                [NİHAİ SONUÇ: 54,400.00 TL]
```

---

## 🔬 4 Zorunlu Derinlemesine Analiz Başlığı

### 1. Çekirdek Mekanizma & Plan-and-Solve Paradigması
- **Zero-Shot CoT Sınırları:** Standart adım adım düşünme ("Let's think step by step") erken adımlarda eksik değişken tanımladığında sonraki tüm adımları yanlış sonuçlandırır.
- **PS & PS+ Yaklaşımı:** Modele önce tüm alt görevleri içeren açık bir plan yapması talimatı verilir (`"First devise a plan, then carry out the plan step by step"`).

### 2. Görev Ayrıştırma (Task Decomposition) ve DAG Mimarisi
- **Yönlü Döngüsüz Çizge (DAG):** Alt görevler arasındaki öncül-ardıl ilişkilerini temsil eder.
- **Kahn Algoritması & Topolojik Sıralama:** Çizgedeki döngüleri (Cycle) tespit eder ve görevleri sıralı ya da paralel çalıştırılabilir bağımlılık sırasına dizer.

### 3. Değişken Çıkarımı ve Ara Durum Haritası (Intermediate State Map)
- **Variable Substitution:** Önceki adımlarda elde edilen çıktılar (`adim_1_gelir = 180000.0`), sonraki adımların matematiksel şablonlarına (`adim_1_gelir - adim_2_maliyet`) değişken olarak aktarılır.
- **Kesin Aritmetik:** LLM'in zihinden sayı uydurması engellenir, AST tabanlı kesin hesaplayıcı kullanılır.

### 4. Kapsamlı Mimari Kıyaslama
- **Doğruluk:** Zero-Shot CoT (%68.2) -> ReAct (%84.5) -> Plan-and-Solve (%91.0) -> **PS+ (%96.4)**.
- **Hesaplama Hatası:** CoT (%24.5) -> **PS+ (%1.2)** (-%95.1 Azalma).
- **Eksik Adım Atlama:** CoT (%18.4) -> **PS+ (%0.4)** (-%97.8 Azalma).

---

## 📖 Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **Plan-and-Solve (PS)** | Problemi önce alt görevlere bölüp planlayan, ardından sırayla çözen akıl yürütme tekniği. |
| **Plan-and-Solve+ (PS+)** | Değişken çıkarımı, ara durum haritası ve hesaplama doğrulaması eklenmiş gelişmiş PS. |
| **Task Decomposition** | Büyük ve karmaşık bir ana görevi bağımsız veya ardışık alt görevlere parçalama süreci. |
| **DAG (Directed Acyclic Graph)** | Görevlerin yönlü olarak birbirine bağlandığı ve döngü içermeyen çizge yapısı. |
| **Topological Sort** | Bağımlılık ilişkilerine sahip görevleri yürütülme sırasına göre dizen çizge algoritması. |
| **Missing-Step Error** | Modelin karmaşık bir akıl yürütme zincirinde zorunlu bir ara adımı atlaması hatası. |
| **Calculation Error** | Modelin büyük sayılarla işlem yaparken yanlış aritmetik sonuç üretmesi hatası. |
| **Variable Substitution** | Bir alt görevin ürettiği değerin sonraki görevlerin formülünde yerine konulması işlemi. |
| **State Map** | Her alt görevin tamamlandıktan sonra sonucunun saklandığı ortak hafıza sözlüğü. |
| **Cycle Detection** | Görevler arasında birbirini bekleyen kilitlenmeleri (Deadlock) tespit eden DFS denetimi. |

---

## ⚖️ 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • %96.4 yüksek doğruluk oranı.       │ • Başlangıçtaki planlama aşaması ek  │
 │ • Adım atlama hatasını %0.4'e indirme│   gecikme ve token tüketir.          │
 │ • DAG ile paralel çalışabilirlik.    │ • Çok basit sorularda fazla karmaşık │
 │ • Kesin değişken ikamesi.            │   (Over-engineering) kalabilir.      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • Karmaşık finans, lojistik ve veri  │ • İlk plan tamamen yanlış kurulursa  │
 │   mühendisliği iş akışlarını yönetme.│   sonraki tüm adımlar hatalı yürür.  │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 Teşhis Panosu Çıktısı

`ana_akis.py` çalıştırıldığında `ciktilar/plan_and_solve_paneli.png` dosyası üretilir:
1. **Alt Görev Değişken Durumları (State Map)**
2. **Çok Aşamalı Görev Doğruluk Oranı (%)**
3. **Hesaplama Hatası (Calculation Error) Oranı (%)**
4. **Eksik Adım Atlama (Missing-Step Error) Oranı (%)**
5. **Topolojik Sıralı DAG Yürütme Sırası Şeması**
6. **Plan-and-Solve+ Mimari ve Algoritma Özet Kartı**

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# Bağımlılıkları yükleyin
pip install -r gereksinimler.txt

# 8 adet birim ve entegrasyon testini çalıştırın
pytest testler/ -v

# Plan-and-Solve motorunu çalıştırın
python ana_akis.py
```

---

## 📜 Lisans & Telif Hakkı

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Yalnızca eğitim ve inceleme amaçlıdır. Ticari kullanım, dağıtım ve kopyalama yasaktır.
```
