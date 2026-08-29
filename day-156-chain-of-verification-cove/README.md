# Day 156: Chain of Verification (CoVe) ile Halüsinasyon Önleme & Fakt Kontrol Motoru

[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg?style=flat-square)](testler/)
[![Phase](https://img.shields.io/badge/FAZ%208-Reasoning%20LLMs-blueviolet.svg?style=flat-square)](../HAFIZA_MUFREDAT_YOL_HARITASI.md)

Bu proje; Meta AI tarafından geliştirilen ve LLM'lerin ürettiği metinlerdeki olgusal (factual) halüsinasyonları kendi kendine ürettiği bağımsız çapraz sorularla denetleyip düzelten **Chain of Verification (CoVe)** mimarisini sıfırdan hayata geçirmektedir.

---

## 🌟 1. Stajyer Seviyesinde Anlaşılır Kılavuz

### ❓ "Chain of Verification (CoVe)" Nedir ve Neden Standart Yanıtlardan Daha Güvenilirdir?
- **Sorun (Doğrudan Yanıt Halüsinasyonu):**
  Bir LLM'e karmaşık bir biyografik soru sorduğunuzda model akıcı bir dille yanıt yazar; fakat metnin içine yanlış bir doğum yeri (İstanbul yerine Ankara) veya yanlış bir kabul yılı (1921 yerine 1923) gibi ince halüsinasyonlar sıkıştırabilir.
- **Çözüm (4 Aşamalı CoVe Boru Hattı):**
  1. **Aşama 1 (Taslak Üretimi):** Model ilk taslak cevabını üretir.
  2. **Aşama 2 (Doğrulama Soruları Planlama):** Taslaktaki iddiaları teyit etmek için tarafsız çapraz sorular üretilir (örn: *"Mehmet Akif hangi şehirde doğdu?"*).
  3. **Aşama 3 (Bağımsız Doğrulama):** Sorular ilk taslağın önyargısından (confirmation bias) etkilenmemesi için izole olarak yanıtlanır.
  4. **Aşama 4 (Düzeltilmiş Yanıt Sentezi):** Çelişkiler ayıklanarak %100 doğrulanmış nihai yanıt oluşturulur!

```
        CHAIN OF VERIFICATION (CoVe) PIPELINE
  [1. Kullanıcı Sorgusu]
           │
           ▼
  [Aşama 1: İlk Taslak Üretimi (Baseline Draft)]
           │
           ▼
  [Aşama 2: Doğrulama Soruları Planlama (Planner)]
    - Mehmet Akif hangi şehirde doğdu?
    - İstiklal Marşı hangi dergahta yazıldı?
           │
           ▼
  [Aşama 3: Bağımsız Fakt Kontrolü (Independent)]
    (Taslak bağlamı verilmeden izole yanıtlanır)
           │
           ▼
  [Aşama 4: Çapraz Kontrol & Düzeltilmiş Yanıt]
```

---

## 🔬 2. 4 Zorunlu Derinlemesine Teknik ve Matematiksel Analiz

### A. Çekirdek Mekanizma: Bağımsız Doğrulama Koşulu (Fact-Conditioning)
- İlk taslak $y_0 = \text{LLM}(x)$ üretildikten sonra iddialar $C = \{c_1, \dots, c_k\}$ ayrıştırılır.
- Her iddia için tarafsız soru $q_i = \text{Plan}(c_i)$ üretilir.
- Bağımsız cevap $a_i = \text{LLM}(q_i \mid \text{No } y_0 \text{ context})$ olarak hesaplanır (Önyargı engelleme).

### B. Onaylama Yanlılığı (Confirmation Bias) İzolasyonu
- Eğer doğrulama sorusu taslağın içine gömülerek sorulursa model kendi hatasını savunma eğilimine girer ($P(c_i \mid y_0, q_i) > P(c_i \mid q_i)$). CoVe, bağlamı sıfırlayarak bu hatayı yok eder.

### C. Çok Aşamalı Fakt Sentezi ve Çelişki Giderme
- $\text{Conflict}(c_i, a_i) \implies$ taslaktaki hatalı varlık $c_i$, doğrulanmış varlık $a_i$ ile yer değiştirir.

### D. Halüsinasyon Temizleme Verimliliği
- 3/3 halüsinasyon içeren tarihsel taslak yanıt, CoVe döngüsü sonucunda %100 doğrulanmış olgusal yanıta dönüştürülür.

---

## 📖 3. Kapsamlı Terimler Sözlüğü (10+ Terim)

| Terim | Tanım |
|:---|:---|
| **CoVe (Chain of Verification)** | LLM çıktılarındaki halüsinasyonları kendi ürettiği sorularla doğrulayan Meta AI yöntemi. |
| **Factual Hallucination** | Modelin gerçek dünyada var olmayan veya yanlış tarih/kişi/olay uydurması. |
| **Baseline Draft** | Modelin herhangi bir doğrulama süzgecinden geçmemiş ilk ham taslak yanıtı. |
| **Verification Question** | Taslaktaki belirli bir iddiayı tarafsızca sınamak için üretilen çapraz kontrol sorusu. |
| **Confirmation Bias (Onay Yanlılığı)** | Modelin kendi yazdığı ilk cevabı haklı çıkarmak için yanlışta ısrar etmesi. |
| **Independent Execution** | Doğrulama sorularının taslak metni görmeden izole bağlamda cevaplanması. |
| **Fact-Checking** | Üretilen iddiaların güvenilir olgusal veritabanı veya parametrik hafıza ile teyidi. |
| **Conflict Resolution** | Taslak ile doğrulama cevabı arasındaki uyuşmazlığın doğru bilgi lehine çözülmesi. |
| **Self-Correction** | Harici bir insan müdahalesi olmadan modelin kendi hatalarını düzeltebilmesi. |
| **Post-Hoc Verification** | Çıkarım tamamlandıktan sonra yapılan doğrulama ve düzeltme aşaması. |

---

## ⚖️ 4. 4 Kutuplu SWOT Matrisi

```
       GÜÇLÜ YÖNLER (STRENGTHS)              ZAYIF YÖNLER (WEAKNESSES)
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 │ • Tarih, tıp ve hukuk alanındaki     │ • 4 aşamalı ek çıkarım adımları      │
 │   halüsinasyonları %90+ azaltma.     │   nedeniyle artan token maliyeti.    │
 │ • Modelin kendi kendini denetleme    │ • Modelin parametrik hafızasında hiç │
 │   (Self-Correction) yeteneği.        │   olmayan bilgilerde sınırılık.      │
 ├──────────────────────────────────────┼──────────────────────────────────────┤
 │ • RAG ve dış arama motorlarıyla     │ • Doğrulama sorusunun kendisinin     │
 │   hibritleştirilerek sıfır hatalı    │   yanlış planlanması halinde         │
 │   otonom araştırma ajanları.         │   oluşabilecek zincirleme hata.      │
 └──────────────────────────────────────┴──────────────────────────────────────┘
       FIRSATLAR (OPPORTUNITIES)               TEHDİTLER (THREATS)
```

---

## 📊 5. Çıktı Panosu

Kod çalıştırıldığında oluşturulan teşhis panosu: `ciktilar/chain_of_verification_cove_paneli.png`

---

## 📜 Lisans

```text
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
```
